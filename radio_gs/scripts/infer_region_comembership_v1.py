#!/usr/bin/env python3
"""Run a frozen RegionCoMembershipV1 checkpoint on a feature authority."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.models.region_comembership_v1 import (
    PAIR_FEATURE_NAMES,
    RegionCoMembershipV1,
)
from radio_gs.scripts.materialize_region_comembership_features_v1 import (
    SCHEMA as FEATURE_SCHEMA,
    _target_execution_authorized,
)
from radio_gs.scripts.train_source_region_comembership_v1 import (
    CHECKPOINT_SCHEMA,
    TRAINING_CONTRACT_SHA256,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.region_comembership_inference_authority.v1"


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_json_sha256(
        {name: tensor_sha256(value) for name, value in sorted(state.items())}
    )


def _validate_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "training_contract",
        "training_contract_sha256",
        "execution_authority",
        "feature_names",
        "model_state_dict",
        "model_state_dict_sha256",
        "selected_epoch",
        "epoch_zero_validation_topology",
        "selected_validation",
        "selected_validation_topology",
        "promotion_gate",
        "selected_probability_threshold",
        "threshold_selection",
        "source_access",
        "target_execution_performed",
    }
    value = dict(checkpoint)
    state = value.get("model_state_dict")
    if (
        set(value) != required
        or value.get("schema") != CHECKPOINT_SCHEMA
        or value.get("schema_version") != 1
        or value.get("training_contract_sha256") != TRAINING_CONTRACT_SHA256
        or value.get("training_contract_sha256")
        != canonical_json_sha256(value.get("training_contract"))
        or value.get("feature_names") != list(PAIR_FEATURE_NAMES)
        or value.get("target_execution_performed") is not False
        or not isinstance(state, Mapping)
        or value.get("model_state_dict_sha256") != _state_sha(state)
        or not isinstance(value.get("selected_epoch"), int)
        or value.get("selected_epoch") < 0
    ):
        raise ValueError("RegionCoMembership checkpoint identity differs")
    return value


def _validate_target_checkpoint_chain(
    *,
    feature: Mapping[str, Any],
    checkpoint_record: Mapping[str, str],
    checkpoint: Mapping[str, Any],
) -> None:
    execution_record = feature.get("target_execution_authority")
    execution_path = validate_file_record(
        execution_record, label="target feature execution authority"
    )
    verified_execution = _target_execution_authorized(
        execution_path,
        expected_sha256=execution_record["sha256"],
        scene_id=str(feature["scene_id"]),
    )
    if verified_execution != dict(execution_record):
        raise ValueError("target feature execution authority record differs")
    execution, _, _ = load_json_object(
        execution_path,
        expected_sha256=execution_record["sha256"],
        label="target feature execution authority",
    )
    result_record = execution.get("four_plus_two_result")
    result_path = validate_file_record(result_record, label="4+2 pilot result")
    result, _, _ = load_json_object(
        result_path,
        expected_sha256=result_record["sha256"],
        label="4+2 pilot result",
    )
    promotion = result.get("promotion_gate")
    selected_topology = result.get("selected_validation_topology")
    checkpoint_topology = checkpoint.get("selected_validation_topology")
    if (
        result.get("schema") != "radio_gs.region_comembership_v1_pilot_result.v1"
        or result.get("status") != "source_only_4train_2validation_pilot_complete"
        or result.get("target_execution_performed") is not False
        or result.get("checkpoint") != dict(checkpoint_record)
        or not isinstance(promotion, Mapping)
        or promotion.get("passed") is not True
        or promotion.get("selected_epoch_positive") is not True
        or promotion.get("selected_topology_score_strictly_exceeds_epoch_zero")
        is not True
        or float(promotion.get("selected_topology_score", float("-inf")))
        <= float(promotion.get("epoch_zero_topology_score", float("inf")))
        or result.get("automatic_epoch_zero_fallback") is not False
        or int(result.get("selected_epoch", -1)) <= 0
        or int(checkpoint.get("selected_epoch", -1)) != int(result["selected_epoch"])
        or checkpoint.get("promotion_gate") != promotion
        or checkpoint_topology != selected_topology
        or checkpoint.get("threshold_selection") != result.get("threshold_selection")
        or float(checkpoint.get("selected_probability_threshold", -1))
        != float(selected_topology["selected"]["threshold"])
    ):
        raise ValueError("target checkpoint is not the promoted 4+2 result checkpoint")


def _feature_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "schema_version",
        "scene_id",
        "domain",
        "producer",
        "target_execution_authority",
        "input_authority",
        "candidate_policy",
        "feature_names",
        "feature_names_sha256",
        "source_access",
    )
    return {key: value[key] for key in keys}


def validate_feature_authority(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "scene_id",
        "domain",
        "producer",
        "target_execution_authority",
        "input_authority",
        "candidate_policy",
        "feature_names",
        "feature_names_sha256",
        "source_access",
        "content_authority_sha256",
        "region_fingerprints",
        "canonical_region_indices",
        "region_rows",
        "token_mask",
        "pair_indices",
        "pair_features",
        "channel_sha256",
        "audit",
    }
    value = dict(payload)
    if (
        set(value) != required
        or value.get("schema") != FEATURE_SCHEMA
        or value.get("schema_version") != 1
        or value.get("domain") not in {"source_parity", "target"}
        or value.get("feature_names") != list(PAIR_FEATURE_NAMES)
        or value.get("feature_names_sha256")
        != canonical_json_sha256(list(PAIR_FEATURE_NAMES))
        or value.get("content_authority_sha256")
        != canonical_json_sha256(_feature_identity(value))
        or (value.get("domain") == "target")
        != (value.get("target_execution_authority") is not None)
    ):
        raise ValueError("RegionCoMembership feature authority identity differs")
    validate_file_record(value["producer"], label="feature authority producer")
    canonical = torch.as_tensor(value["canonical_region_indices"])
    rows = torch.as_tensor(value["region_rows"])
    mask = torch.as_tensor(value["token_mask"])
    pairs = torch.as_tensor(value["pair_indices"])
    features = torch.as_tensor(value["pair_features"])
    count = int(canonical.numel())
    if (
        canonical.dtype != torch.int64
        or canonical.ndim != 1
        or rows.ndim != 2
        or rows.shape[0] != count
        or rows.dtype not in {torch.int32, torch.int64}
        or mask.dtype != torch.bool
        or mask.shape != rows.shape
        or pairs.dtype != torch.int64
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or features.dtype != torch.float32
        or features.shape != (pairs.shape[1], len(PAIR_FEATURE_NAMES))
        or bool((pairs < 0).any())
        or bool((pairs >= count).any())
        or bool((pairs[0] >= pairs[1]).any())
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("RegionCoMembership feature tensors differ")
    channels = value.get("channel_sha256")
    if not isinstance(channels, Mapping):
        raise ValueError("RegionCoMembership feature channel hashes differ")
    for name in (
        "canonical_region_indices",
        "region_rows",
        "token_mask",
        "pair_indices",
        "pair_features",
    ):
        if channels.get(name) != tensor_sha256(value[name]):
            raise ValueError(f"RegionCoMembership feature channel changed: {name}")
    return value


def infer_probabilities(
    feature_payload: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> tuple[torch.Tensor, float]:
    feature = validate_feature_authority(feature_payload)
    checkpoint = _validate_checkpoint(checkpoint)
    state = checkpoint["model_state_dict"]
    required_state = {
        "feature_median",
        "feature_robust_scale",
        "logit.weight",
        "logit.bias",
    }
    if set(state) != required_state:
        raise ValueError("RegionCoMembership checkpoint state differs")
    model = RegionCoMembershipV1(state["feature_median"], state["feature_robust_scale"])
    model.load_state_dict(state, strict=True)
    threshold = float(checkpoint["selected_probability_threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("RegionCoMembership checkpoint threshold differs")
    model.eval()
    with torch.no_grad():
        probability = model.probability(feature["pair_features"]).cpu().contiguous()
    return probability, threshold


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"RegionCoMembership inference already exists: {output}")
    feature, feature_sha, feature_path = load_torch_mapping(
        args.feature_authority,
        expected_sha256=args.expected_feature_authority_sha256,
        map_location="cpu",
        label="RegionCoMembership feature authority",
    )
    feature = validate_feature_authority(feature)
    checkpoint, checkpoint_sha, checkpoint_path = load_torch_mapping(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        map_location="cpu",
        label="RegionCoMembership checkpoint",
    )
    checkpoint_record = {"path": str(checkpoint_path), "sha256": checkpoint_sha}
    if feature["domain"] == "target":
        _validate_target_checkpoint_chain(
            feature=feature,
            checkpoint_record=checkpoint_record,
            checkpoint=checkpoint,
        )
    probability, threshold = infer_probabilities(feature, checkpoint)
    selected = probability >= threshold
    identity = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene_id": feature["scene_id"],
        "domain": feature["domain"],
        "producer": file_record(Path(__file__).resolve()),
        "feature_authority": {"path": str(feature_path), "sha256": feature_sha},
        "checkpoint": checkpoint_record,
        "probability_threshold": threshold,
        "source_access": feature["source_access"],
    }
    payload = {
        **identity,
        "content_authority_sha256": canonical_json_sha256(identity),
        "region_fingerprints": list(feature["region_fingerprints"]),
        "canonical_region_indices": feature["canonical_region_indices"],
        "pair_indices": feature["pair_indices"],
        "pair_probabilities": probability,
        "accepted_edge_mask": selected.contiguous(),
        "channel_sha256": {},
        "audit": {
            "canonical_regions": int(feature["canonical_region_indices"].numel()),
            "candidate_pairs": int(probability.numel()),
            "accepted_edges": int(selected.sum()),
        },
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name])
        for name in (
            "canonical_region_indices",
            "pair_indices",
            "pair_probabilities",
            "accepted_edge_mask",
        )
    }
    written = write_torch_noclobber(output, payload)
    return {
        "status": "region_comembership_checkpoint_inference_complete",
        "output": file_record(written),
        "audit": payload["audit"],
        "target_metric_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-authority", required=True)
    parser.add_argument("--expected-feature-authority-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
