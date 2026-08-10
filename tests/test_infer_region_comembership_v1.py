from __future__ import annotations

from pathlib import Path

import pytest
import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.models.region_comembership_v1 import (
    PAIR_FEATURE_NAMES,
    RegionCoMembershipV1,
)
from radio_gs.scripts import infer_region_comembership_v1 as inference
from radio_gs.scripts import materialize_region_comembership_features_v1 as materializer
from radio_gs.scripts import train_source_region_comembership_v1 as trainer
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    write_frozen_json,
)


def _feature() -> dict:
    identity = {
        "schema": materializer.SCHEMA,
        "schema_version": 1,
        "scene_id": "scene0001_00",
        "domain": "source_parity",
        "producer": file_record(Path(materializer.__file__).resolve()),
        "target_execution_authority": None,
        "input_authority": {},
        "candidate_policy": {
            "descriptor_neighbors": 16,
            "centroid_neighbors": 16,
            "anchor_support_edges": True,
        },
        "feature_names": list(PAIR_FEATURE_NAMES),
        "feature_names_sha256": canonical_json_sha256(list(PAIR_FEATURE_NAMES)),
        "source_access": materializer._source_access("source_parity"),
    }
    payload = {
        **identity,
        "content_authority_sha256": canonical_json_sha256(identity),
        "region_fingerprints": ["a", "b", "c"],
        "canonical_region_indices": torch.arange(3, dtype=torch.int64),
        "region_rows": torch.arange(3, dtype=torch.int64)[:, None],
        "token_mask": torch.ones(3, 1, dtype=torch.bool),
        "pair_indices": torch.tensor([[0, 1], [1, 2]], dtype=torch.int64),
        "pair_features": torch.zeros(2, len(PAIR_FEATURE_NAMES)),
        "channel_sha256": {},
        "audit": {},
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name])
        for name in (
            "canonical_region_indices",
            "region_rows",
            "token_mask",
            "pair_indices",
            "pair_features",
        )
    }
    return payload


def test_epoch_zero_checkpoint_inference_is_half_probability() -> None:
    model = RegionCoMembershipV1(torch.zeros(15), torch.ones(15))
    state = model.state_dict()
    topology = {
        "selected": {
            "threshold": 0.55,
            "scene_macro_instance_macro": {
                "topology_score": 0.4,
                "iou": 0.5,
                "f1": 0.6,
                "contamination": 0.1,
                "giant_excess": 0.0,
            },
        }
    }
    promotion = {
        "selected_epoch_positive": True,
        "selected_topology_score_strictly_exceeds_epoch_zero": True,
        "epoch_zero_topology_score": 0.2,
        "selected_topology_score": 0.4,
        "passed": True,
    }
    contract = trainer.training_contract()
    checkpoint = {
        "schema": trainer.CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "training_contract": contract,
        "training_contract_sha256": trainer.TRAINING_CONTRACT_SHA256,
        "execution_authority": {"path": "/frozen", "sha256": "a" * 64},
        "feature_names": list(PAIR_FEATURE_NAMES),
        "model_state_dict": state,
        "model_state_dict_sha256": inference._state_sha(state),
        "selected_epoch": 1,
        "epoch_zero_validation_topology": {},
        "selected_validation": {},
        "selected_validation_topology": topology,
        "promotion_gate": promotion,
        "selected_probability_threshold": 0.55,
        "threshold_selection": {},
        "source_access": trainer.source_access(),
        "target_execution_performed": False,
    }
    probability, threshold = inference.infer_probabilities(_feature(), checkpoint)
    assert probability.tolist() == [0.5, 0.5]
    assert threshold == 0.55


def test_target_chain_rejects_checkpoint_other_than_promoted_result(
    tmp_path: Path,
) -> None:
    expected_checkpoint = tmp_path / "expected.pt"
    expected_checkpoint.write_bytes(b"expected")
    result = {
        "schema": "radio_gs.region_comembership_v1_pilot_result.v1",
        "status": "source_only_4train_2validation_pilot_complete",
        "target_execution_performed": False,
        "checkpoint": file_record(expected_checkpoint),
        "promotion_gate": {
            "passed": True,
            "selected_epoch_positive": True,
            "selected_topology_score_strictly_exceeds_epoch_zero": True,
            "selected_topology_score": 0.4,
            "epoch_zero_topology_score": 0.2,
        },
        "automatic_epoch_zero_fallback": False,
        "selected_epoch": 3,
        "selected_validation_topology": {"selected": {"threshold": 0.65}},
        "threshold_selection": {},
    }
    result_path = write_frozen_json(tmp_path / "result.json", result)
    execution_path = write_frozen_json(
        tmp_path / "execution.json",
        {
            "schema": materializer.TARGET_EXECUTION_SCHEMA,
            "schema_version": 1,
            "status": "authorized_after_topology_4plus2_promotion",
            "scene_id": "scene0001_00",
            "target_gate": file_record(
                Path(materializer.__file__).resolve().parents[2]
                / materializer.TARGET_GATE
            ),
            "four_plus_two_result": file_record(result_path),
            "target_feature_materialization_authorized": True,
            "target_checkpoint_inference_authorized": True,
            "target_metric_authorized": False,
        },
    )
    feature = _feature()
    feature["domain"] = "target"
    feature["target_execution_authority"] = file_record(execution_path)
    checkpoint = {
        "selected_epoch": 3,
        "promotion_gate": result["promotion_gate"],
        "selected_validation_topology": {"selected": {"threshold": 0.65}},
        "selected_probability_threshold": 0.65,
        "threshold_selection": {},
    }
    wrong_checkpoint = tmp_path / "wrong.pt"
    wrong_checkpoint.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="promoted 4\\+2"):
        inference._validate_target_checkpoint_chain(
            feature=feature,
            checkpoint_record=file_record(wrong_checkpoint),
            checkpoint=checkpoint,
        )
