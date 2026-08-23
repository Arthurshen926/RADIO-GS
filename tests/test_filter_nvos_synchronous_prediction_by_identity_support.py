from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from radio_gs.scripts.build_nvos_identity_component_threshold_authority import (
    ARTIFACT_TYPE as THRESHOLD_AUTHORITY_TYPE,
)
from radio_gs.scripts.filter_nvos_synchronous_prediction_by_identity_support import (
    FILTERED_RECEIPT_TYPE,
    filter_batch,
)
from radio_gs.scripts.render_nvos_synchronous_candidate_marginal import RECEIPT_TYPE


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_batch_filter_is_delete_only_and_binds_frozen_authority(tmp_path) -> None:
    parent_probability = np.zeros((8, 12), dtype=np.float32)
    parent_probability[1:4, 1:4] = 1
    parent_probability[4:7, 9:11] = 1
    parent_path = tmp_path / "parent.npy"
    np.save(parent_path, parent_probability, allow_pickle=False)
    parent_receipt_path = tmp_path / "parent.json"
    _json(
        parent_receipt_path,
        {
            "artifact_type": RECEIPT_TYPE,
            "scene_id": "fern",
            "target_frame_id": "target",
            "prediction": {"path": str(parent_path), "sha256": _sha256(parent_path)},
            "prediction_sealed_before_target_ground_truth": True,
            "target_mask_opened": False,
            "target_metric_opened": False,
            "threshold": 0.5,
        },
    )

    signed = np.full((4, 6), -1, dtype=np.float32)
    signed[0:2, 0:2] = 1
    signed_path = tmp_path / "signed.npy"
    np.save(signed_path, signed, allow_pickle=False)
    unary_path = tmp_path / "unary.json"
    _json(
        unary_path,
        {
            "kind": "promptable_nvs_continuous_score_predictions",
            "prediction_root": ".",
            "predictions": {"fern": {"target": signed_path.name}},
            "prediction_sha256": {"fern": {"target": _sha256(signed_path)}},
            "safety": {"evaluation_ground_truth_opened": False},
        },
    )
    authority_path = tmp_path / "authority.json"
    _json(
        authority_path,
        {
            "artifact_type": THRESHOLD_AUTHORITY_TYPE,
            "selected_threshold": 0.05,
            "target_mask_opened": False,
            "target_metric_opened": False,
        },
    )
    output = tmp_path / "filtered"
    report = filter_batch(
        SimpleNamespace(
            receipt=[str(parent_receipt_path)],
            require_full8=False,
            signed_unary_manifest=str(unary_path),
            expected_signed_unary_manifest_sha256=_sha256(unary_path),
            threshold_authority=str(authority_path),
            expected_threshold_authority_sha256=_sha256(authority_path),
            minimum_local_identity_density=0.05,
            authority_bound_replay_after_prior_metrics=False,
            output_root=str(output),
        )
    )

    receipt = json.loads(
        (output / "fern/prediction_receipt.json").read_text(encoding="utf-8")
    )
    filtered = np.load(output / "fern/target_probability.npy", allow_pickle=False) >= 0.5
    assert report["all_outputs_sealed"] is True
    assert receipt["artifact_type"] == FILTERED_RECEIPT_TYPE
    assert receipt["adds_foreground"] is False
    assert not bool((filtered & ~(parent_probability >= 0.5)).any())
    assert bool(filtered[1:4, 1:4].all())
    assert not bool(filtered[4:7, 9:11].any())
