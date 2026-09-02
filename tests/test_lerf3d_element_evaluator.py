import json
from pathlib import Path

import numpy as np
import pytest
import torch

from radio_gs.evaluation.lerf3d_element_evaluator import (
    AGGREGATION,
    COHORT_SCHEMA,
    POSTERIOR_API,
    QUERY_UNIT_SEMANTICS,
    READY_STATUS,
    TARGET_AUTHORITY_STATUS,
    TARGET_DOMAIN,
    TARGET_SCHEMA,
    CarrierBinding,
    ElementPredictionBatch,
    ElementQueryKey,
    ElementTargetAuthority,
    Lerf3DElementContractError,
    PredictionInformationPolicy,
    _mask_sha256,
    compose_element_posterior,
    load_cohort_manifest,
    load_prediction_batch,
    load_target_authority,
    score_element_predictions,
    seal_prediction_batch,
    select_element_mask,
    validate_prediction_batch,
)
from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.object_memory import SparseObjectAssignments
from radio_gs.v4.query import QueryPacket


REPOSITORY = Path(__file__).resolve().parents[1]
FAKE_DIGEST = "a" * 64


def _ready_manifest(tmp_path, scene_queries):
    scenes = [
        {
            "scene_id": scene_id,
            "expected_query_count": len(query_ids),
            "query_ids": sorted(query_ids),
        }
        for scene_id, query_ids in scene_queries
    ]
    payload = {
        "schema": COHORT_SCHEMA,
        "protocol_id": "synthetic_element_contract_test",
        "status": READY_STATUS,
        "target_domain": TARGET_DOMAIN,
        "query_unit_semantics": QUERY_UNIT_SEMANTICS,
        "aggregation": AGGREGATION,
        "metrics": {
            "iou": "binary_element_iou",
            "strict_accuracy_thresholds": [0.25, 0.5],
        },
        "scenes": scenes,
        "expected_total_query_count": sum(len(row[1]) for row in scene_queries),
        "target_authority_schema": TARGET_SCHEMA,
    }
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return load_cohort_manifest(path)


def _prediction(manifest, masks, bindings):
    return ElementPredictionBatch(
        cohort_manifest_sha256=manifest.sha256,
        masks=masks,
        carrier_bindings=bindings,
        method_id="synthetic_method",
        selection_threshold=0.6,
        query_selection_mode="multi_instance",
        information_policy=PredictionInformationPolicy(),
        posterior_api=POSTERIOR_API,
    )


def _targets(manifest, masks, bindings):
    return ElementTargetAuthority(
        cohort_manifest_sha256=manifest.sha256,
        masks=masks,
        carrier_bindings=bindings,
        association_authority={
            "authority_id": "synthetic_official_contract_test",
            "release": "test-only",
            "source_sha256": "b" * 64,
            "official_element_domain": True,
        },
        status=TARGET_AUTHORITY_STATUS,
    )


def test_repository_manifest_records_real_element_target_blocker():
    manifest = load_cohort_manifest(
        REPOSITORY
        / "paper/artifacts/lerf3d_element_target_cohort_contract_audit_20260831.json"
    )
    assert manifest.ready is False
    assert len(manifest.scenes) == 4
    assert manifest.expected_total_query_count == 208
    with pytest.raises(Lerf3DElementContractError, match="formal LERF3D scoring is blocked"):
        manifest.require_ready()


def test_element_posterior_uses_only_shared_mixture_sum_api():
    assignments = SparseObjectAssignments(
        token_ids=torch.tensor([[0, 1], [0, 1]]),
        weights=torch.tensor([[0.4, 0.3], [0.8, 0.1]]),
        unknown_weight=torch.tensor([0.3, 0.1]),
        num_tokens=2,
    )
    query = QueryPacket("single_instance")
    result = compose_element_posterior(
        assignments,
        query,
        torch.tensor([0.6, 0.3]),
        null_probability=0.1,
    )
    # Both retained token contributions are summed. A max compositor gives .24/.48.
    assert result.foreground.tolist() == pytest.approx([0.33, 0.51])
    selected = select_element_mask(
        assignments,
        query,
        torch.tensor([0.6, 0.3]),
        null_probability=0.1,
        threshold=0.3,
    )
    assert selected.tolist() == [True, True]


def test_object_element_adapter_rejects_local_semantic_query():
    assignments = SparseObjectAssignments.from_dense(torch.tensor([[1.0]]), top_k=1)
    with pytest.raises(ValueError, match="local surface semantic memory"):
        compose_element_posterior(
            assignments,
            QueryPacket("local_semantic"),
            torch.tensor([1.0]),
        )


def test_element_metrics_use_strict_thresholds_and_scene_equal_macro(tmp_path):
    manifest = _ready_manifest(
        tmp_path,
        [("scene_a", ["q025", "q050"]), ("scene_b", ["q100"])],
    )
    bindings = {
        "scene_a": CarrierBinding(FAKE_DIGEST, 4),
        "scene_b": CarrierBinding(FAKE_DIGEST, 4),
    }
    targets = {
        ElementQueryKey("scene_a", "q025"): np.array([1, 1, 1, 1]),
        ElementQueryKey("scene_a", "q050"): np.array([1, 1, 0, 0]),
        ElementQueryKey("scene_b", "q100"): np.array([1, 0, 0, 0]),
    }
    predictions = {
        ElementQueryKey("scene_a", "q025"): np.array([1, 0, 0, 0]),
        ElementQueryKey("scene_a", "q050"): np.array([1, 0, 0, 0]),
        ElementQueryKey("scene_b", "q100"): np.array([1, 0, 0, 0]),
    }
    report = score_element_predictions(
        _prediction(manifest, predictions, bindings),
        _targets(manifest, targets, bindings),
        manifest,
    )

    assert [row["iou"] for row in report["per_query"]] == pytest.approx([0.25, 0.5, 1.0])
    assert [row["acc025"] for row in report["per_query"]] == [False, True, True]
    assert [row["acc050"] for row in report["per_query"]] == [False, False, True]
    assert report["scene_equal_macro"] == pytest.approx(
        {"miou": 0.6875, "acc025": 0.75, "acc050": 0.5}
    )
    assert report["target_domain"] == TARGET_DOMAIN
    assert report["uses_2d_polygon_proxy"] is False
    assert report["formal_lerf3d_eligible"] is True


def test_prediction_inventory_is_total_and_cannot_open_targets(tmp_path):
    manifest = _ready_manifest(tmp_path, [("scene", ["one", "two"])])
    bindings = {"scene": CarrierBinding(FAKE_DIGEST, 2)}
    incomplete = _prediction(
        manifest,
        {ElementQueryKey("scene", "one"): np.array([1, 0])},
        bindings,
    )
    with pytest.raises(Lerf3DElementContractError, match="inventory differs"):
        validate_prediction_batch(incomplete, manifest)

    opened = ElementPredictionBatch(
        **{
            **vars(
                _prediction(
                    manifest,
                    {
                        ElementQueryKey("scene", "one"): np.array([1, 0]),
                        ElementQueryKey("scene", "two"): np.array([0, 1]),
                    },
                    bindings,
                )
            ),
            "information_policy": PredictionInformationPolicy(
                target_membership_opened=True
            ),
        }
    )
    with pytest.raises(Lerf3DElementContractError, match="forbidden target information"):
        validate_prediction_batch(opened, manifest)


def test_prediction_receipt_round_trip_is_hash_bound(tmp_path):
    manifest = _ready_manifest(tmp_path, [("scene", ["query"])])
    bindings = {"scene": CarrierBinding(FAKE_DIGEST, 3)}
    batch = _prediction(
        manifest,
        {ElementQueryKey("scene", "query"): np.array([1, 0, 1])},
        bindings,
    )
    archive = tmp_path / "predictions.npz"
    receipt = tmp_path / "predictions.json"
    seal_prediction_batch(
        batch, manifest, archive_path=archive, receipt_path=receipt
    )
    loaded = load_prediction_batch(receipt, manifest)
    assert loaded.masks[ElementQueryKey("scene", "query")].tolist() == [True, False, True]

    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(Lerf3DElementContractError, match="archive SHA256 differs"):
        load_prediction_batch(receipt, manifest)


def test_target_loader_accepts_only_one_dimensional_element_authority(tmp_path):
    manifest = _ready_manifest(tmp_path, [("scene", ["query"])])
    mask = np.array([1, 0, 1], dtype=np.uint8)
    archive = tmp_path / "targets.npz"
    with archive.open("wb") as handle:
        np.savez_compressed(handle, target_000000=mask)
    payload = {
        "schema": TARGET_SCHEMA,
        "status": TARGET_AUTHORITY_STATUS,
        "target_domain": TARGET_DOMAIN,
        "cohort_manifest_sha256": manifest.sha256,
        "association_authority": {
            "authority_id": "synthetic_official_contract_test",
            "release": "test-only",
            "source_sha256": "b" * 64,
            "official_element_domain": True,
        },
        "archive": {"path": str(archive), "sha256": sha256_file(archive)},
        "carrier_bindings": {
            "scene": {"sha256": FAKE_DIGEST, "num_elements": 3}
        },
        "targets": [
            {
                "scene_id": "scene",
                "query_id": "query",
                "array_key": "target_000000",
                "shape": [3],
                "mask_sha256": _mask_sha256(mask.astype(bool)),
            }
        ],
    }
    receipt = tmp_path / "targets.json"
    receipt.write_text(json.dumps(payload, sort_keys=True))
    authority = load_target_authority(receipt, manifest)
    assert authority.masks[ElementQueryKey("scene", "query")].tolist() == [True, False, True]

    payload["polygon_annotations"] = []
    receipt.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(Lerf3DElementContractError, match="2-D polygon/raster proxy"):
        load_target_authority(receipt, manifest)


def test_target_loader_rejects_two_dimensional_masks(tmp_path):
    manifest = _ready_manifest(tmp_path, [("scene", ["query"])])
    mask = np.ones((2, 2), dtype=np.uint8)
    archive = tmp_path / "targets_2d.npz"
    with archive.open("wb") as handle:
        np.savez_compressed(handle, target=mask)
    receipt = tmp_path / "targets_2d.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": TARGET_SCHEMA,
                "status": TARGET_AUTHORITY_STATUS,
                "target_domain": TARGET_DOMAIN,
                "cohort_manifest_sha256": manifest.sha256,
                "association_authority": {
                    "authority_id": "synthetic",
                    "release": "test-only",
                    "source_sha256": "b" * 64,
                    "official_element_domain": True,
                },
                "archive": {"path": str(archive), "sha256": sha256_file(archive)},
                "carrier_bindings": {
                    "scene": {"sha256": FAKE_DIGEST, "num_elements": 4}
                },
                "targets": [
                    {
                        "scene_id": "scene",
                        "query_id": "query",
                        "array_key": "target",
                        "shape": [4],
                        "mask_sha256": "0" * 64,
                    }
                ],
            }
        )
    )
    with pytest.raises(Lerf3DElementContractError, match="one-dimensional"):
        load_target_authority(receipt, manifest)
