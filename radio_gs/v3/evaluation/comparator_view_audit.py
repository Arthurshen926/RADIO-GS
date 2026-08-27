"""Fail-closed view-lineage audit for historical source comparators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from radio_gs.v3.training.instance_upper_bound import sha256_file


def _split_frame_ids(
    source_records: Sequence[Mapping[str, Any]],
    *,
    train_residues: Sequence[int],
    dev_residue: int,
    audit_residue: int,
) -> dict[str, list[int]]:
    splits: dict[str, set[int]] = {"train": set(), "dev": set(), "audit": set()}
    train_residue_set = {int(value) for value in train_residues}
    for record in source_records:
        view = int(record["source_view_index"])
        frame = int(record["frame_id"])
        residue = view % 4
        if residue in train_residue_set:
            splits["train"].add(frame)
        if residue == int(dev_residue):
            splits["dev"].add(frame)
        if residue == int(audit_residue):
            splits["audit"].add(frame)
    return {name: sorted(values) for name, values in splits.items()}


def audit_historical_comparator_views(
    payload: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    *,
    train_residues: Sequence[int] = (1, 2),
    dev_residue: int = 3,
    audit_residue: int = 0,
) -> dict[str, Any]:
    """Prove whether a historical field is held out from the current source split.

    This function deliberately raises on incomplete lineage. A comparator without
    verifiable construction views cannot be described as held out.
    """

    metadata = payload.get("mpr_cache_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("historical comparator lacks mpr_cache_metadata lineage")
    manifest_value = metadata.get("feature_frame_manifest")
    expected_sha256 = metadata.get("feature_frame_manifest_sha256")
    selected = metadata.get("selected_dataset_indices")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ValueError("historical comparator lacks feature_frame_manifest")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("historical comparator lacks a hash-bound feature manifest")
    if not isinstance(selected, (list, tuple)) or not selected:
        raise ValueError("historical comparator lacks selected_dataset_indices")

    manifest_path = Path(manifest_value).resolve(strict=True)
    actual_sha256 = sha256_file(manifest_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("historical comparator feature manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("historical comparator feature manifest lacks frames")

    selected_frame_ids: list[int] = []
    for raw_index in selected:
        index = int(raw_index)
        if index < 0 or index >= len(frames):
            raise ValueError("historical selected_dataset_indices is out of range")
        frame = frames[index]
        if not isinstance(frame, Mapping) or "frame_idx" not in frame:
            raise ValueError("historical feature manifest frame lacks frame_idx")
        selected_frame_ids.append(int(frame["frame_idx"]))

    current = _split_frame_ids(
        source_records,
        train_residues=train_residues,
        dev_residue=dev_residue,
        audit_residue=audit_residue,
    )
    selected_set = set(selected_frame_ids)
    overlap = {
        name: sorted(selected_set.intersection(frame_ids))
        for name, frame_ids in current.items()
    }
    strictly_heldout = len(overlap["dev"]) == 0
    return {
        "schema": "radio_gs.sugm_v3.historical_comparator_view_audit.v1",
        "status": (
            "strictly_heldout_comparator"
            if strictly_heldout
            else "diagnostic_nonheldout_comparator"
        ),
        "strictly_heldout": strictly_heldout,
        "eligible_as_heldout_gate": strictly_heldout,
        "manifest": {"path": str(manifest_path), "sha256": actual_sha256},
        "historical_selected_dataset_indices": [int(value) for value in selected],
        "historical_selected_frame_ids": sorted(set(selected_frame_ids)),
        "current_split_frame_ids": current,
        "overlap_frame_ids": overlap,
        "overlap_counts": {name: len(values) for name, values in overlap.items()},
    }
