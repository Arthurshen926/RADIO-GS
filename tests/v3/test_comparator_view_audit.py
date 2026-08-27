import hashlib
import json

import pytest

from radio_gs.v3.evaluation.comparator_view_audit import (
    audit_historical_comparator_views,
)


def _payload(tmp_path, frame_ids=(10, 20, 30, 40), selected=(0, 2)):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"frames": [{"frame_idx": value} for value in frame_ids]}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "mpr_cache_metadata": {
            "feature_frame_manifest": str(path),
            "feature_frame_manifest_sha256": digest,
            "selected_dataset_indices": list(selected),
        }
    }


def _records():
    return [
        {"source_view_index": 1, "frame_id": 10},
        {"source_view_index": 3, "frame_id": 30},
        {"source_view_index": 0, "frame_id": 40},
    ]


def test_audit_marks_dev_overlap_nonheldout(tmp_path):
    report = audit_historical_comparator_views(_payload(tmp_path), _records())
    assert report["status"] == "diagnostic_nonheldout_comparator"
    assert report["eligible_as_heldout_gate"] is False
    assert report["overlap_frame_ids"] == {"train": [10], "dev": [30], "audit": []}


def test_audit_accepts_verified_disjoint_dev(tmp_path):
    report = audit_historical_comparator_views(
        _payload(tmp_path, selected=(0, 3)), _records()
    )
    assert report["status"] == "strictly_heldout_comparator"
    assert report["eligible_as_heldout_gate"] is True


@pytest.mark.parametrize("mutation", ["missing", "bad_hash", "out_of_range"])
def test_audit_fails_closed_on_unverifiable_lineage(tmp_path, mutation):
    payload = _payload(tmp_path)
    metadata = payload["mpr_cache_metadata"]
    if mutation == "missing":
        del metadata["feature_frame_manifest"]
    elif mutation == "bad_hash":
        metadata["feature_frame_manifest_sha256"] = "0" * 64
    else:
        metadata["selected_dataset_indices"] = [99]
    with pytest.raises(ValueError):
        audit_historical_comparator_views(payload, _records())
