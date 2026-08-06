import hashlib
import json
from copy import deepcopy

import pytest
import torch

from radio_gs.querying.query_specific_propagation_cv import (
    ACTION_SOURCE_UNARY,
    ACTION_SURFACE_SAFE_PROPAGATED,
    evaluate_source_observation_oof_artifacts,
    prepare_source_observation_oof_fold,
)
from radio_gs.querying.source_observation_authority import (
    seal_or_load_source_observation_evidence_authority,
)
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _write_source_observation_oof_artifact,
    _write_source_observation_oof_gate_receipt,
)


def _source_evidence(count: int = 900):
    rows = torch.arange(count)
    valid = torch.ones(count, dtype=torch.bool)
    positive = torch.where(rows % 2 == 0, 0.8, 0.0)
    negative = torch.where(rows % 2 == 1, 0.7, 0.0)
    raw_positive = positive * 2.0
    raw_negative = negative * 3.0
    return rows, valid, positive, negative, raw_positive, raw_negative


def _tensor_sha256(value: torch.Tensor) -> str:
    array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _fold_payloads():
    rows, valid, positive, negative, raw_positive, raw_negative = _source_evidence()
    labels = rows % 2 == 0
    unary = torch.where(labels, 0.65, 0.35)
    propagated = torch.where(labels, 0.9, 0.1)
    payloads = {}
    for heldout_fold in range(3):
        authority = prepare_source_observation_oof_fold(
            rows,
            valid,
            positive,
            negative,
            raw_positive,
            raw_negative,
            heldout_fold=heldout_fold,
        )
        tensors = {
            "valid": valid,
            "global_rows": rows,
            "fold_ids": authority.fold_ids,
            "observed": authority.observed,
            "heldout": authority.heldout,
            "signed_reference_evidence": authority.signed_reference_evidence,
            "reference_weight": authority.reference_weight,
            "unary_probability": unary,
            "surface_safe_propagated_probability": propagated,
        }
        payloads[heldout_fold] = {
            "artifact_type": "source_observation_surface_safe_oof_fold_v1",
            "scene_id": "fern",
            "protocol_hash": "protocol",
            "heldout_fold": heldout_fold,
            "num_folds": 3,
            "method_contract_sha256": "a" * 64,
            "capability_cache_sha256": "b" * 64,
            "support_graph_sha256": "c" * 64,
            "source_evidence_authority_sha256": "d" * 64,
            "source_evidence_authority_content_sha256": "e" * 64,
            "tensor_sha256": {
                name: _tensor_sha256(value) for name, value in tensors.items()
            },
            "heldout_prompt_evidence_after_clear": {
                "positive_weight_sum": 0.0,
                "negative_weight_sum": 0.0,
                "raw_positive_mass_sum": 0.0,
                "raw_negative_mass_sum": 0.0,
            },
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_computed": False,
            **tensors,
        }
    return payloads


def test_prepare_source_observation_fold_clears_every_compiler_input():
    rows, valid, positive, negative, raw_positive, raw_negative = _source_evidence()
    authority = prepare_source_observation_oof_fold(
        rows,
        valid,
        positive,
        negative,
        raw_positive,
        raw_negative,
        heldout_fold=1,
    )
    for value in (
        authority.training_positive_weight,
        authority.training_negative_weight,
        authority.training_raw_positive_mass,
        authority.training_raw_negative_mass,
    ):
        assert bool((value[authority.heldout] == 0).all())
    torch.testing.assert_close(
        authority.signed_reference_evidence,
        raw_positive - raw_negative,
    )
    torch.testing.assert_close(
        authority.reference_weight,
        raw_positive + raw_negative,
    )


def test_source_observation_gate_uses_only_oof_rows_and_selects_propagation():
    payloads = _fold_payloads()
    result = evaluate_source_observation_oof_artifacts(payloads)
    assert result.selected_action == ACTION_SURFACE_SAFE_PROPAGATED
    assert (
        result.metrics[ACTION_SURFACE_SAFE_PROPAGATED][
            "responsibility_balanced_log_loss"
        ]
        < result.metrics[ACTION_SOURCE_UNARY][
            "responsibility_balanced_log_loss"
        ]
    )
    assert bool(
        torch.isfinite(
            result.oof_predictions[ACTION_SURFACE_SAFE_PROPAGATED][
                result.observed
            ]
        ).all()
    )


@pytest.mark.parametrize("tamper", ["prediction", "fold", "leakage", "target"])
def test_source_observation_gate_fails_closed_on_fold_tamper(tamper):
    payloads = _fold_payloads()
    payloads = {fold: deepcopy(payload) for fold, payload in payloads.items()}
    if tamper == "prediction":
        payloads[1]["unary_probability"][10] += 0.01
    elif tamper == "fold":
        payloads[1]["heldout"][10] = ~payloads[1]["heldout"][10]
        payloads[1]["tensor_sha256"]["heldout"] = _tensor_sha256(
            payloads[1]["heldout"]
        )
    elif tamper == "leakage":
        payloads[1]["heldout_prompt_evidence_after_clear"][
            "raw_positive_mass_sum"
        ] = 1e-6
    else:
        payloads[1]["target_mask_opened"] = True
    with pytest.raises((ValueError, RuntimeError)):
        evaluate_source_observation_oof_artifacts(payloads)


def test_fold_writer_and_gate_receipt_are_immutable_and_target_blind(tmp_path):
    capability = tmp_path / "capability.pt"
    graph = tmp_path / "graph.pt"
    capability.write_bytes(b"capability")
    graph.write_bytes(b"graph")
    output = tmp_path / "oof"
    rows, valid, positive, negative, raw_positive, raw_negative = _source_evidence()
    labels = rows % 2 == 0
    unary = torch.where(labels, 0.65, 0.35)
    propagated = torch.where(labels, 0.9, 0.1)
    evidence_authority = seal_or_load_source_observation_evidence_authority(
        output / "source_observation_evidence_authority.pt",
        heldout_fold=0,
        provenance={"fixed": True},
        valid=valid,
        global_rows=rows,
        positive_weight=positive,
        negative_weight=negative,
        raw_positive_mass=raw_positive,
        raw_negative_mass=raw_negative,
    )
    for heldout_fold in range(3):
        authority = prepare_source_observation_oof_fold(
            rows,
            valid,
            positive,
            negative,
            raw_positive,
            raw_negative,
            heldout_fold=heldout_fold,
        )
        _write_source_observation_oof_artifact(
            output / f"fold_{heldout_fold}.pt",
            scene_id="fern",
            protocol_hash="protocol",
            heldout_fold=heldout_fold,
            capability_cache=capability,
            support_graph=graph,
            authority=authority,
            evidence_authority=evidence_authority,
            valid=valid,
            global_rows=rows,
            unary_probability=unary,
            propagated_probability=propagated,
            method_contract={"fixed": True},
        )
    receipt_path, receipt = _write_source_observation_oof_gate_receipt(output)
    assert receipt_path is not None
    assert receipt["selected_action"] == ACTION_SURFACE_SAFE_PROPAGATED
    assert receipt["target_rgb_opened"] is False
    assert receipt["target_mask_opened"] is False
    assert receipt["target_metric_computed"] is False
    assert (
        receipt["source_evidence_authority_sha256"]
        == evidence_authority.sha256
    )
    assert json.loads(receipt_path.read_text())["fold_artifacts"].keys() == {
        "0",
        "1",
        "2",
    }
    second_path, second_receipt = _write_source_observation_oof_gate_receipt(output)
    assert second_path == receipt_path
    assert second_receipt == receipt
