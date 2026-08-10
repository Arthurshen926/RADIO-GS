from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pytest
import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import region_comembership_v2_formal as formal
from radio_gs.models.region_comembership_v2 import PAIR_FEATURE_NAMES, RegionCoMembershipV2
from radio_gs.scripts import infer_region_comembership_v2 as inference
from radio_gs.scripts import materialize_region_comembership_features_v2 as materializer
from radio_gs.scripts import train_source_region_comembership_v2 as trainer
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


def _metric(iou: float, f1: float, selected_regions: float) -> dict[str, float]:
    return {
        "iou": iou,
        "f1": f1,
        "contamination": 0.0,
        "giant_excess": 0.0,
        "selected_units": selected_regions * 10.0,
        "selected_regions": selected_regions,
        "topology_score": iou,
    }


def _candidate(
    *, epoch: int, maximum_regions: int, threshold: float, metric: dict[str, float]
) -> dict:
    per_scene = {scene: dict(metric) for scene in trainer.VALIDATION_SCENES}
    return {
        "epoch": epoch,
        "method": trainer.METHODS[0],
        "maximum_regions": maximum_regions,
        "threshold": threshold,
        "scene_macro": dict(metric),
        "per_scene": per_scene,
    }


def _source_execution(tmp_path: Path) -> Path:
    records = {}
    for scene in (*trainer.TRAIN_SCENES, *trainer.VALIDATION_SCENES):
        path = tmp_path / f"{scene}.source.pt"
        path.write_bytes(scene.encode())
        records[scene] = file_record(path)
    authority = {
        "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_complete_v2_4train_2validation_preflight",
        "implementation": file_record(Path(trainer.__file__).resolve()),
        "preregistration": file_record(
            Path(trainer.__file__).resolve().parents[2] / trainer.PREREGISTRATION
        ),
        "efficiency_addendum": file_record(
            Path(trainer.__file__).resolve().parents[2] / trainer.EFFICIENCY_ADDENDUM
        ),
        "source_train": [
            {"scene_id": scene, "authority": records[scene]}
            for scene in trainer.TRAIN_SCENES
        ],
        "source_validation": [
            {"scene_id": scene, "authority": records[scene]}
            for scene in trainer.VALIDATION_SCENES
        ],
        "training_authorized": True,
        "target_execution_authorized": False,
        "source_access": trainer.source_access(),
    }
    return write_frozen_json(tmp_path / "source_execution.json", authority)


def _formal_chain(tmp_path: Path, *, promoted: bool = True) -> tuple[Path, Path]:
    singleton_metric = _metric(0.25, 0.4, 1.0)
    selected_metric = _metric(0.5, 0.6 if promoted else 0.4, 2.0)
    singleton = _candidate(
        epoch=0,
        maximum_regions=1,
        threshold=max(trainer.THRESHOLDS),
        metric=singleton_metric,
    )
    selected = _candidate(
        epoch=25,
        maximum_regions=2,
        threshold=0.5,
        metric=selected_metric,
    )
    flags = {
        "selected_epoch_positive": True,
        "topology_strictly_exceeds_singleton": True,
        "iou_strictly_exceeds_singleton": True,
        "f1_strictly_exceeds_singleton": promoted,
    }
    promotion = {
        **flags,
        "singleton": singleton_metric,
        "selected": selected_metric,
        "passed": all(flags.values()),
    }
    model = RegionCoMembershipV2(torch.zeros(21), torch.ones(21))
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checkpoint = {
        "schema": trainer.CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "training_contract": trainer.training_contract(),
        "training_contract_sha256": canonical_json_sha256(trainer.training_contract()),
        "execution_authority": file_record(_source_execution(tmp_path)),
        "feature_names": list(PAIR_FEATURE_NAMES),
        "normalization": {
            "median": state["feature_median"],
            "robust_scale": state["feature_robust_scale"],
        },
        "model_state_dict": state,
        "model_state_dict_sha256": canonical_json_sha256(
            {name: tensor_sha256(value) for name, value in sorted(state.items())}
        ),
        "selected_epoch": 25,
        "selected_rule": {
            "method": selected["method"],
            "maximum_regions": selected["maximum_regions"],
            "threshold": selected["threshold"],
        },
        "selected_validation": selected,
        "singleton_validation": singleton,
        "promotion_gate": promotion,
        "source_access": trainer.source_access(),
        "target_execution_performed": False,
    }
    checkpoint_path = write_torch_noclobber(tmp_path / "checkpoint.pt", checkpoint)
    result = {
        "schema": formal.RESULT_SCHEMA,
        "schema_version": 1,
        "status": "source_only_v2_4train_2validation_complete",
        "checkpoint": file_record(checkpoint_path),
        "selected_validation": selected,
        "singleton_validation": singleton,
        "promotion_gate": promotion,
        "exact_candidate_count": 2,
        "exact_candidates": [singleton, selected],
        "proxy_audit": {str(epoch): {} for epoch in trainer.SNAPSHOT_EPOCHS},
        "history": [
            {"epoch": epoch, "train_scene_macro_balanced_bce": 1.0 / epoch}
            for epoch in range(1, trainer.EPOCHS + 1)
        ],
        "source_access": trainer.source_access(),
        "target_execution_performed": False,
    }
    result_path = write_frozen_json(tmp_path / "checkpoint.pt.json", result)
    return checkpoint_path, result_path


def _target_inputs(tmp_path: Path) -> dict[str, dict[str, str]]:
    result = {}
    for name in formal.TARGET_INPUT_NAMES:
        path = tmp_path / f"{name}.target"
        path.write_bytes(name.encode())
        result[name] = file_record(path)
    return result


def _target_execution(
    tmp_path: Path,
    *,
    checkpoint_path: Path,
    result_path: Path,
    inputs: dict[str, dict[str, str]],
    feature_output: Path,
    inference_output: Path,
) -> Path:
    root = Path(trainer.__file__).resolve().parents[2]
    authority = {
        "schema": formal.TARGET_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": formal.TARGET_EXECUTION_STATUS,
        "scene_id": "synthetic_target",
        "preregistration": file_record(root / trainer.PREREGISTRATION),
        "efficiency_addendum": file_record(root / trainer.EFFICIENCY_ADDENDUM),
        "four_plus_two_result": file_record(result_path),
        "promoted_checkpoint": file_record(checkpoint_path),
        "target_feature_inputs": inputs,
        "target_feature_output": str(feature_output.resolve()),
        "target_inference_output": str(inference_output.resolve()),
        "target_feature_materialization_authorized": True,
        "target_checkpoint_inference_authorized": True,
        "target_metric_authorized": False,
        "access_audit": {
            "benchmark_images_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "target_metrics_computed": False,
        },
    }
    return write_frozen_json(tmp_path / "target_execution.json", authority)


def _feature(
    tmp_path: Path,
    *,
    domain: str,
    inputs: dict[str, dict[str, str]],
    execution_path: Path | None,
) -> dict:
    identity = {
        "schema": materializer.SCHEMA,
        "schema_version": materializer.SCHEMA_VERSION,
        "scene_id": "synthetic_target" if domain == "target" else "scene0001_00",
        "domain": domain,
        "producer": file_record(Path(materializer.__file__).resolve()),
        "target_execution_authority": (
            file_record(execution_path) if execution_path is not None else None
        ),
        "input_authority": inputs,
        "candidate_policy": {
            "descriptor_neighbors": 16,
            "centroid_neighbors": 16,
            "anchor_support_edges": True,
        },
        "feature_names": list(PAIR_FEATURE_NAMES),
        "feature_names_sha256": canonical_json_sha256(list(PAIR_FEATURE_NAMES)),
        "source_access": materializer._source_access(domain),
    }
    return materializer._finalize_payload(
        identity=identity,
        region_fingerprints=["a", "b", "c"],
        canonical_region_indices=torch.arange(3, dtype=torch.int64),
        region_rows=torch.arange(3, dtype=torch.int64)[:, None],
        token_mask=torch.ones(3, 1, dtype=torch.bool),
        pair_indices=torch.tensor([[0, 1], [1, 2]], dtype=torch.int64),
        pair_features=torch.zeros(2, len(PAIR_FEATURE_NAMES), dtype=torch.float32),
        audit={"canonical_regions": 3, "candidate_pairs": 2, "pair_feature_dimension": 21},
    )


def test_source_inference_uses_formal_checkpoint_and_emits_selected_rule(
    tmp_path: Path,
) -> None:
    checkpoint_path, _ = _formal_chain(tmp_path)
    checkpoint, _, _ = load_torch_mapping(checkpoint_path, map_location="cpu")
    source_inputs = {}
    for name in materializer.SOURCE_INPUT_NAMES:
        path = tmp_path / name
        path.write_bytes(name.encode())
        source_inputs[name] = file_record(path)
    feature = _feature(
        tmp_path, domain="source_parity", inputs=source_inputs, execution_path=None
    )
    probability, rule = inference.infer_probabilities(feature, checkpoint)
    assert torch.equal(probability, torch.full((2,), 0.5))
    assert rule == {"method": "maximum_product", "maximum_regions": 2, "threshold": 0.5}


def test_target_gate_rejects_failed_v2_promotion_before_target_inputs(
    tmp_path: Path,
) -> None:
    checkpoint_path, result_path = _formal_chain(tmp_path, promoted=False)
    missing_inputs = {
        name: {"path": str(tmp_path / f"missing_{name}"), "sha256": "a" * 64}
        for name in formal.TARGET_INPUT_NAMES
    }
    feature_output = tmp_path / "feature.pt"
    inference_output = tmp_path / "inference.pt"
    execution = _target_execution(
        tmp_path,
        checkpoint_path=checkpoint_path,
        result_path=result_path,
        inputs=missing_inputs,
        feature_output=feature_output,
        inference_output=inference_output,
    )
    with pytest.raises(ValueError, match="source promotion did not pass"):
        formal.validate_target_execution_authority(
            execution,
            expected_sha256=file_record(execution)["sha256"],
            scene_id="synthetic_target",
            expected_feature_output=feature_output,
        )


def test_formal_result_rejects_promoted_candidate_that_is_not_global_maximum(
    tmp_path: Path,
) -> None:
    _, result_path = _formal_chain(tmp_path)
    import json

    result = json.loads(result_path.read_text(encoding="utf-8"))
    inferior_metric = _metric(0.4, 0.5, 2.0)
    inferior = _candidate(
        epoch=25, maximum_regions=2, threshold=0.55, metric=inferior_metric
    )
    result["selected_validation"] = inferior
    result["promotion_gate"] = {
        "selected_epoch_positive": True,
        "topology_strictly_exceeds_singleton": True,
        "iou_strictly_exceeds_singleton": True,
        "f1_strictly_exceeds_singleton": True,
        "singleton": result["singleton_validation"]["scene_macro"],
        "selected": inferior_metric,
        "passed": True,
    }
    result["exact_candidates"].append(inferior)
    result["exact_candidate_count"] += 1
    with pytest.raises(ValueError, match="global maximum"):
        formal.validate_result(result, require_promotion=True)


def test_target_inference_binds_feature_checkpoint_and_selected_rule(
    tmp_path: Path,
) -> None:
    checkpoint_path, result_path = _formal_chain(tmp_path)
    inputs = _target_inputs(tmp_path)
    feature_output = tmp_path / "target_feature.pt"
    inference_output = tmp_path / "target_inference.pt"
    execution = _target_execution(
        tmp_path,
        checkpoint_path=checkpoint_path,
        result_path=result_path,
        inputs=inputs,
        feature_output=feature_output,
        inference_output=inference_output,
    )
    feature = _feature(
        tmp_path, domain="target", inputs=inputs, execution_path=execution
    )
    write_torch_noclobber(feature_output, feature)
    result = inference.run(
        argparse.Namespace(
            feature_authority=str(feature_output),
            expected_feature_authority_sha256=file_record(feature_output)["sha256"],
            checkpoint=str(checkpoint_path),
            expected_checkpoint_sha256=file_record(checkpoint_path)["sha256"],
            output=str(inference_output),
        )
    )
    assert result["domain"] == "target"
    assert result["readout_executed"] is False
    assert result["selected_rule"]["maximum_regions"] == 2
    payload, _, _ = load_torch_mapping(inference_output, map_location="cpu")
    inference.validate_inference_authority(payload)
    assert torch.equal(payload["pair_probabilities"], torch.full((2,), 0.5))


def test_target_inference_rejects_same_bytes_at_unpromoted_checkpoint_path(
    tmp_path: Path,
) -> None:
    checkpoint_path, result_path = _formal_chain(tmp_path)
    inputs = _target_inputs(tmp_path)
    feature_output = tmp_path / "target_feature.pt"
    inference_output = tmp_path / "target_inference.pt"
    execution = _target_execution(
        tmp_path,
        checkpoint_path=checkpoint_path,
        result_path=result_path,
        inputs=inputs,
        feature_output=feature_output,
        inference_output=inference_output,
    )
    write_torch_noclobber(
        feature_output,
        _feature(tmp_path, domain="target", inputs=inputs, execution_path=execution),
    )
    substituted = tmp_path / "substituted_checkpoint.pt"
    shutil.copyfile(checkpoint_path, substituted)
    with pytest.raises(ValueError, match="feature/result/checkpoint chain"):
        inference.run(
            argparse.Namespace(
                feature_authority=str(feature_output),
                expected_feature_authority_sha256=file_record(feature_output)["sha256"],
                checkpoint=str(substituted),
                expected_checkpoint_sha256=file_record(substituted)["sha256"],
                output=str(inference_output),
            )
        )


def test_feature_axis_hash_rejects_fingerprint_and_canonical_drift(
    tmp_path: Path,
) -> None:
    source_inputs = {}
    for name in materializer.SOURCE_INPUT_NAMES:
        path = tmp_path / name
        path.write_bytes(name.encode())
        source_inputs[name] = file_record(path)
    feature = _feature(
        tmp_path, domain="source_parity", inputs=source_inputs, execution_path=None
    )
    feature["region_fingerprints"] = ["b", "a", "c"]
    with pytest.raises(ValueError, match="identity|SHA axis"):
        materializer.validate_feature_authority(feature)

    feature = _feature(
        tmp_path, domain="source_parity", inputs=source_inputs, execution_path=None
    )
    feature["canonical_region_indices"] = torch.tensor([0, 2, 1])
    feature["channel_sha256"]["canonical_region_indices"] = tensor_sha256(
        feature["canonical_region_indices"]
    )
    feature["tensor_authority_sha256"] = canonical_json_sha256(feature["channel_sha256"])
    with pytest.raises(ValueError, match="identity|SHA axis"):
        materializer.validate_feature_authority(feature)


def test_inference_authority_rejects_rule_drift_from_bound_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path, _ = _formal_chain(tmp_path)
    source_inputs = {}
    for name in materializer.SOURCE_INPUT_NAMES:
        path = tmp_path / name
        path.write_bytes(name.encode())
        source_inputs[name] = file_record(path)
    feature_path = write_torch_noclobber(
        tmp_path / "source_feature.pt",
        _feature(
            tmp_path,
            domain="source_parity",
            inputs=source_inputs,
            execution_path=None,
        ),
    )
    inference_path = tmp_path / "source_inference.pt"
    inference.run(
        argparse.Namespace(
            feature_authority=str(feature_path),
            expected_feature_authority_sha256=file_record(feature_path)["sha256"],
            checkpoint=str(checkpoint_path),
            expected_checkpoint_sha256=file_record(checkpoint_path)["sha256"],
            output=str(inference_path),
        )
    )
    payload, _, _ = load_torch_mapping(inference_path, map_location="cpu")
    payload["selected_rule"] = {
        **payload["selected_rule"],
        "method": "dual_path_widest",
    }
    payload["content_authority_sha256"] = canonical_json_sha256(
        inference._identity(payload)
    )
    with pytest.raises(ValueError, match="selected rule differs"):
        inference.validate_inference_authority(payload)


def test_target_materializer_refuses_v1_promoted_feature_path() -> None:
    with pytest.raises(ValueError, match="authority-bound inputs"):
        materializer.materialize(
            argparse.Namespace(
                domain="target",
                scene_id="synthetic_target",
                v1_feature_authority="legacy_v1_target.pt",
                expected_v1_feature_authority_sha256="a" * 64,
                capability_descriptor=None,
                expected_capability_descriptor_sha256=None,
                execution_authority="v2_execution.json",
                expected_execution_authority_sha256="b" * 64,
                output="new_v2_target.pt",
            )
        )


def test_target_materializer_validates_promotion_before_target_base_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = False

    def reject_gate(*args, **kwargs):
        raise ValueError("promotion rejected before target")

    def target_base_would_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("target base must remain unopened")

    monkeypatch.setattr(materializer, "validate_target_execution_authority", reject_gate)
    monkeypatch.setattr(materializer, "_load_target_base", target_base_would_open)
    with pytest.raises(ValueError, match="promotion rejected before target"):
        materializer.materialize(
            argparse.Namespace(
                domain="target",
                scene_id="synthetic_target",
                v1_feature_authority=None,
                expected_v1_feature_authority_sha256=None,
                capability_descriptor=None,
                expected_capability_descriptor_sha256=None,
                execution_authority=str(tmp_path / "execution.json"),
                expected_execution_authority_sha256="a" * 64,
                output=str(tmp_path / "feature.pt"),
            )
        )
    assert opened is False
