import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import radio_gs.querying.query_specific_propagation_cv as propagation_cv
from radio_gs.querying.query_specific_propagation_cv import (
    ACTION_SURFACE_SAFE_PROPAGATED,
    evaluate_source_observation_footprint_oof_artifacts,
    prepare_source_observation_footprint_oof_fold,
)
from radio_gs.querying.source_footprint_fold_authority import (
    FIELD_BASE_ACTION,
    MINIMUM_CLASS_ROWS,
    build_source_raster_dominant_footprint_authority,
    save_source_footprint_fold_authority,
    splitmix64_source_group_folds,
)
from radio_gs.querying.source_observation_authority import (
    seal_or_load_source_observation_evidence_authority,
)
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _load_source_observation_oof_deployment_gate,
    _load_source_observation_footprint_authority,
    _write_source_observation_footprint_field_base_receipt,
    _write_source_observation_footprint_oof_artifact,
    _write_source_observation_footprint_oof_gate_receipt,
)


TRIPLET_SHA256 = "a" * 64


def _tensor_sha256(value: torch.Tensor) -> str:
    array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _balanced_source_footprint():
    height = width = 16
    block_folds = splitmix64_source_group_folds(torch.arange(64))
    blocks = [int(torch.where(block_folds == fold)[0][0]) for fold in range(3)]
    rows = torch.arange(1, 1 + 6 * MINIMUM_CLASS_ROWS)
    pixel_ids = []
    primitive_ids = []
    positive_local = torch.zeros(rows.numel(), dtype=torch.bool)
    cursor = 0
    for block in blocks:
        block_y, block_x = divmod(block, 8)
        positive_pixel = (2 * block_y) * width + 2 * block_x
        negative_pixel = positive_pixel + 1
        for _ in range(MINIMUM_CLASS_ROWS):
            pixel_ids.append(positive_pixel)
            primitive_ids.append(int(rows[cursor]))
            positive_local[cursor] = True
            cursor += 1
        for _ in range(MINIMUM_CLASS_ROWS):
            pixel_ids.append(negative_pixel)
            primitive_ids.append(int(rows[cursor]))
            cursor += 1
    authority = build_source_raster_dominant_footprint_authority(
        torch.tensor(pixel_ids),
        torch.tensor(primitive_ids),
        torch.full((len(pixel_ids),), 1.0 / MINIMUM_CLASS_ROWS),
        height=height,
        width=width,
        hierarchy_primitive_rows=rows,
        primitive_id_domain="global_rows",
        source_triplet_authority_sha256=TRIPLET_SHA256,
        expected_source_triplet_authority_sha256=TRIPLET_SHA256,
    )
    valid = torch.zeros(int(rows[-1]) + 2, dtype=torch.bool)
    valid[rows] = True
    positive = torch.zeros(valid.shape, dtype=torch.float32)
    negative = torch.zeros_like(positive)
    positive[rows] = torch.where(positive_local, 0.8, 0.0)
    negative[rows] = torch.where(positive_local, 0.0, 0.7)
    # Invalid rows deliberately contain evidence. Structured OOF is authorized
    # to clear only the four tensors on held-out valid footprint rows.
    positive[~valid] = 0.13
    negative[~valid] = 0.17
    raw_positive = positive * 2.0
    raw_negative = negative * 3.0
    return authority, rows, valid, positive, negative, raw_positive, raw_negative


def _prepared_payloads(authority, rows, valid, positive, negative, raw_positive, raw_negative):
    labels = raw_positive > raw_negative
    unary = torch.where(labels, 0.65, 0.35)
    propagated = torch.where(labels, 0.9, 0.1)
    payloads = {}
    for heldout_fold in range(3):
        fold, decision = prepare_source_observation_footprint_oof_fold(
            authority,
            rows,
            valid,
            positive,
            negative,
            raw_positive,
            raw_negative,
            heldout_fold=heldout_fold,
            expected_footprint_authority_sha256=authority.authority_sha256,
        )
        assert fold is not None and decision.run_source_oof
        tensors = {
            "valid": valid,
            "global_rows": rows,
            "fold_ids": fold.fold_ids,
            "observed": fold.observed,
            "heldout": fold.heldout,
            "signed_reference_evidence": fold.signed_reference_evidence,
            "reference_weight": fold.reference_weight,
            "population_positive_weight": positive,
            "population_negative_weight": negative,
            "unary_probability": unary,
            "surface_safe_propagated_probability": propagated,
        }
        payloads[heldout_fold] = {
            "artifact_type": "source_observation_surface_safe_footprint_oof_fold_v1",
            "fold_assignment": "splitmix64_source_footprint_group_v1",
            "scene_id": "lego",
            "protocol_hash": "protocol",
            "heldout_fold": heldout_fold,
            "num_folds": 3,
            "method_contract_sha256": "b" * 64,
            "capability_cache_sha256": "c" * 64,
            "support_graph_sha256": "d" * 64,
            "source_evidence_authority_sha256": "e" * 64,
            "source_evidence_authority_content_sha256": "f" * 64,
            "source_footprint_fold_authority": "/frozen/footprint.pt",
            "source_footprint_fold_authority_file_sha256": "1" * 64,
            "source_footprint_fold_authority_sha256": authority.authority_sha256,
            "source_footprint_fold_authority_tensor_bundle_sha256": (
                authority.tensor_bundle_sha256
            ),
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


def test_structured_prepare_clears_whole_groups_and_only_valid_four_evidence():
    authority, rows, valid, positive, negative, raw_positive, raw_negative = (
        _balanced_source_footprint()
    )
    fold, decision = prepare_source_observation_footprint_oof_fold(
        authority,
        rows,
        valid,
        positive,
        negative,
        raw_positive,
        raw_negative,
        heldout_fold=1,
        expected_footprint_authority_sha256=authority.authority_sha256,
    )
    assert fold is not None and decision.run_source_oof
    expected_local = splitmix64_source_group_folds(authority.group_ids) == 1
    expected_full = torch.zeros_like(valid)
    expected_full[rows] = expected_local
    assert torch.equal(fold.heldout, expected_full)
    for training, original in zip(
        (
            fold.training_positive_weight,
            fold.training_negative_weight,
            fold.training_raw_positive_mass,
            fold.training_raw_negative_mass,
        ),
        (positive, negative, raw_positive, raw_negative),
    ):
        assert bool((training[expected_full] == 0).all())
        assert torch.equal(training[~valid], original[~valid])


def test_structured_prepare_requires_exact_capability_valid_and_footprint_rows():
    authority, rows, valid, positive, negative, raw_positive, raw_negative = (
        _balanced_source_footprint()
    )
    changed_valid = valid.clone()
    changed_valid[0] = True
    with pytest.raises(ValueError, match="must match exactly"):
        prepare_source_observation_footprint_oof_fold(
            authority,
            rows,
            changed_valid,
            positive,
            negative,
            raw_positive,
            raw_negative,
            heldout_fold=0,
            expected_footprint_authority_sha256=authority.authority_sha256,
        )


def test_structured_gate_uses_group_authority_and_never_calls_row_cv(monkeypatch):
    fixture = _balanced_source_footprint()
    authority, rows, valid, positive, negative, raw_positive, raw_negative = fixture
    payloads = _prepared_payloads(*fixture)

    def forbidden_row_cv(*args, **kwargs):
        raise AssertionError("legacy row CV must not run")

    monkeypatch.setattr(propagation_cv, "audit_signed_cv_population", forbidden_row_cv)
    result = evaluate_source_observation_footprint_oof_artifacts(
        payloads,
        footprint_authority=authority,
        footprint_authority_path="/frozen/footprint.pt",
        footprint_authority_file_sha256="1" * 64,
    )
    assert result.selected_action == ACTION_SURFACE_SAFE_PROPAGATED
    assert len(result.fold_reports) == 3

    tampered = {fold: deepcopy(payload) for fold, payload in payloads.items()}
    tampered[1]["fold_ids"][rows[0]] = (tampered[1]["fold_ids"][rows[0]] + 1) % 3
    tampered[1]["tensor_sha256"]["fold_ids"] = _tensor_sha256(
        tampered[1]["fold_ids"]
    )
    with pytest.raises(ValueError, match="fold ids|invariant"):
        evaluate_source_observation_footprint_oof_artifacts(
            tampered,
            footprint_authority=authority,
            footprint_authority_path="/frozen/footprint.pt",
            footprint_authority_file_sha256="1" * 64,
        )


def test_structured_writer_and_gate_bind_footprint_lineage(tmp_path):
    fixture = _balanced_source_footprint()
    authority, rows, valid, positive, negative, raw_positive, raw_negative = fixture
    footprint_path = tmp_path / "footprint.pt"
    footprint_artifact = save_source_footprint_fold_authority(
        authority, footprint_path
    )
    capability = tmp_path / "capability.pt"
    graph = tmp_path / "graph.pt"
    capability.write_bytes(b"capability")
    graph.write_bytes(b"graph")
    output = tmp_path / "oof"
    evidence = seal_or_load_source_observation_evidence_authority(
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
    labels = raw_positive > raw_negative
    unary = torch.where(labels, 0.65, 0.35)
    propagated = torch.where(labels, 0.9, 0.1)
    for heldout_fold in range(3):
        fold, decision = prepare_source_observation_footprint_oof_fold(
            authority,
            rows,
            valid,
            positive,
            negative,
            raw_positive,
            raw_negative,
            heldout_fold=heldout_fold,
            expected_footprint_authority_sha256=authority.authority_sha256,
        )
        assert fold is not None and decision.run_source_oof
        _write_source_observation_footprint_oof_artifact(
            output / f"fold_{heldout_fold}.pt",
            scene_id="lego",
            protocol_hash="protocol",
            heldout_fold=heldout_fold,
            capability_cache=capability,
            support_graph=graph,
            authority=fold,
            evidence_authority=evidence,
            footprint_path=footprint_path,
            footprint_file_sha256=footprint_artifact["file_sha256"],
            footprint_authority=authority,
            valid=valid,
            global_rows=rows,
            population_positive_weight=positive,
            population_negative_weight=negative,
            unary_probability=unary,
            propagated_probability=propagated,
            method_contract={"fixed": True},
        )
    gate_path, receipt = _write_source_observation_footprint_oof_gate_receipt(
        output
    )
    assert gate_path is not None
    assert receipt["artifact_type"] == (
        "source_observation_surface_safe_footprint_oof_gate_v1"
    )
    assert receipt["fold_assignment"] == "splitmix64_source_footprint_group_v1"
    assert receipt["source_footprint_fold_authority_sha256"] == (
        authority.authority_sha256
    )
    assert receipt["selected_action"] == ACTION_SURFACE_SAFE_PROPAGATED


def test_degenerate_structured_population_seals_field_base(tmp_path):
    rows = torch.arange(4)
    authority = build_source_raster_dominant_footprint_authority(
        torch.arange(4),
        rows,
        torch.ones(4),
        height=8,
        width=8,
        hierarchy_primitive_rows=rows,
        primitive_id_domain="global_rows",
        source_triplet_authority_sha256=TRIPLET_SHA256,
        expected_source_triplet_authority_sha256=TRIPLET_SHA256,
    )
    valid = torch.ones(4, dtype=torch.bool)
    positive = torch.tensor([1.0, 0.0, 1.0, 0.0])
    negative = 1.0 - positive
    fold, decision = prepare_source_observation_footprint_oof_fold(
        authority,
        rows,
        valid,
        positive,
        negative,
        positive,
        negative,
        heldout_fold=0,
        expected_footprint_authority_sha256=authority.authority_sha256,
    )
    assert fold is None and decision.selected_action == FIELD_BASE_ACTION
    footprint_path = tmp_path / "footprint.pt"
    footprint_artifact = save_source_footprint_fold_authority(
        authority, footprint_path
    )
    capability = tmp_path / "capability.pt"
    graph = tmp_path / "graph.pt"
    capability.write_bytes(b"capability")
    graph.write_bytes(b"graph")
    evidence = seal_or_load_source_observation_evidence_authority(
        tmp_path / "oof" / "source_observation_evidence_authority.pt",
        heldout_fold=0,
        provenance={"fixed": True},
        valid=valid,
        global_rows=rows,
        positive_weight=positive,
        negative_weight=negative,
        raw_positive_mass=positive,
        raw_negative_mass=negative,
    )
    path, receipt = _write_source_observation_footprint_field_base_receipt(
        tmp_path / "oof",
        scene_id="lego",
        protocol_hash="protocol",
        capability_cache=capability,
        support_graph=graph,
        evidence_authority=evidence,
        footprint_path=footprint_path,
        footprint_file_sha256=footprint_artifact["file_sha256"],
        footprint_authority=authority,
        population_decision=decision,
        method_contract={"fixed": True},
    )
    assert path.is_file()
    assert receipt["selected_action"] == FIELD_BASE_ACTION
    assert receipt["fold_artifacts"] == {}
    assert receipt["target_mask_opened"] is False


def test_explicit_footprint_loader_requires_all_hashes_and_exact_rows(tmp_path):
    fixture = _balanced_source_footprint()
    authority, rows, valid, *_ = fixture
    path = tmp_path / "footprint.pt"
    artifact = save_source_footprint_fold_authority(authority, path)
    bank = SimpleNamespace(valid=valid, global_rows=rows)
    args = SimpleNamespace(
        source_observation_oof_fold_mode="source_footprint_v1",
        source_observation_oof_output_dir=str(tmp_path / "oof"),
        source_footprint_fold_authority=str(path),
        source_footprint_fold_authority_file_sha256=artifact["file_sha256"],
        source_footprint_fold_authority_sha256=authority.authority_sha256,
    )
    loaded = _load_source_observation_footprint_authority(args, bank)
    assert loaded is not None and loaded[2].authority_sha256 == authority.authority_sha256
    args.source_footprint_fold_authority_file_sha256 = ""
    with pytest.raises(ValueError, match="requires authority path"):
        _load_source_observation_footprint_authority(args, bank)


def test_deployment_gate_binds_selected_action_assets_and_protocol(tmp_path):
    capability = tmp_path / "capability.pt"
    graph = tmp_path / "graph.pt"
    capability.write_bytes(b"capability")
    graph.write_bytes(b"graph")
    file_sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    gate = {
        "schema_version": 1,
        "artifact_type": "source_observation_surface_safe_footprint_oof_gate_v1",
        "scene_id": "lego",
        "protocol_hash": "protocol",
        "method_contract_sha256": "a" * 64,
        "capability_cache_sha256": file_sha(capability),
        "support_graph_sha256": file_sha(graph),
        "selected_action": ACTION_SURFACE_SAFE_PROPAGATED,
        "selection_rule": "source-only",
        "full_fit_predictions_used_as_oof": False,
        "connected_selection": "off",
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    args = SimpleNamespace(
        source_observation_oof_gate_receipt=str(gate_path),
        source_observation_oof_gate_receipt_sha256=file_sha(gate_path),
        canonical_capability_cache=str(capability),
        canonical_support_graph=str(graph),
        registered_readout_stage="propagated",
        registered_selection_mode="all_components",
        query_conditioned_diffusion_kernel="none",
    )
    deployment = _load_source_observation_oof_deployment_gate(
        args, scene_id="lego", protocol_hash="protocol"
    )
    assert deployment is not None
    assert deployment["selected_action"] == ACTION_SURFACE_SAFE_PROPAGATED
    assert deployment["required_readout_stage"] == "propagated"

    args.registered_readout_stage = "unary_prior"
    with pytest.raises(ValueError, match="requires --registered-readout-stage propagated"):
        _load_source_observation_oof_deployment_gate(
            args, scene_id="lego", protocol_hash="protocol"
        )


def test_deployment_gate_fails_closed_on_lineage_or_target_access(tmp_path):
    capability = tmp_path / "capability.pt"
    graph = tmp_path / "graph.pt"
    capability.write_bytes(b"capability")
    graph.write_bytes(b"graph")
    file_sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    gate = {
        "schema_version": 1,
        "artifact_type": "source_observation_surface_safe_footprint_oof_gate_v1",
        "scene_id": "lego",
        "protocol_hash": "protocol",
        "method_contract_sha256": "a" * 64,
        "capability_cache_sha256": "0" * 64,
        "support_graph_sha256": file_sha(graph),
        "selected_action": ACTION_SURFACE_SAFE_PROPAGATED,
        "selection_rule": "source-only",
        "full_fit_predictions_used_as_oof": False,
        "connected_selection": "off",
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    args = SimpleNamespace(
        source_observation_oof_gate_receipt=str(gate_path),
        source_observation_oof_gate_receipt_sha256=file_sha(gate_path),
        canonical_capability_cache=str(capability),
        canonical_support_graph=str(graph),
        registered_readout_stage="propagated",
        registered_selection_mode="all_components",
        query_conditioned_diffusion_kernel="none",
    )
    with pytest.raises(ValueError, match="capability cache differs"):
        _load_source_observation_oof_deployment_gate(
            args, scene_id="lego", protocol_hash="protocol"
        )

    gate["capability_cache_sha256"] = file_sha(capability)
    gate["target_mask_opened"] = True
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    args.source_observation_oof_gate_receipt_sha256 = file_sha(gate_path)
    with pytest.raises(ValueError, match="target-access flag differs"):
        _load_source_observation_oof_deployment_gate(
            args, scene_id="lego", protocol_hash="protocol"
        )
