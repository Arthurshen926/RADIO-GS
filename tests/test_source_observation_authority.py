from pathlib import Path

import pytest
import torch

from radio_gs.querying.source_observation_authority import (
    SOURCE_EVIDENCE_TENSOR_NAMES,
    seal_or_load_source_observation_evidence_authority,
)


def _inputs(count: int = 90) -> dict[str, torch.Tensor]:
    rows = torch.arange(count)
    valid = torch.ones(count, dtype=torch.bool)
    positive = torch.where(rows % 3 == 0, torch.tensor(0.8), torch.tensor(0.0))
    negative = torch.where(rows % 3 != 0, torch.tensor(0.7), torch.tensor(0.0))
    return {
        "valid": valid,
        "global_rows": rows,
        "positive_weight": positive,
        "negative_weight": negative,
        "raw_positive_mass": positive * 11.0,
        "raw_negative_mass": negative * 13.0,
    }


def _provenance() -> dict[str, object]:
    return {
        "scene_id": "fern",
        "protocol_hash": "a" * 64,
        "method_contract_sha256": "b" * 64,
        "capability_cache_sha256": "c" * 64,
        "support_graph_sha256": "d" * 64,
    }


def test_fold_zero_seals_once_and_later_folds_replay_bitwise(tmp_path: Path):
    path = tmp_path / "source_observation_evidence_authority.pt"
    inputs = _inputs()
    first = seal_or_load_source_observation_evidence_authority(
        path,
        heldout_fold=0,
        provenance=_provenance(),
        **inputs,
    )
    jittered = dict(inputs)
    for name in SOURCE_EVIDENCE_TENSOR_NAMES[2:]:
        value = inputs[name]
        jittered[name] = torch.where(value != 0, value * (1.0 + 5e-6), value)
    second = seal_or_load_source_observation_evidence_authority(
        path,
        heldout_fold=1,
        provenance=_provenance(),
        **jittered,
    )
    assert first.sha256 == second.sha256
    assert first.content_sha256 == second.content_sha256
    for name in SOURCE_EVIDENCE_TENSOR_NAMES:
        assert torch.equal(first.tensors[name], second.tensors[name])
    assert max(second.replay_max_relative_error.values()) > 0


def test_later_fold_cannot_create_the_authority(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="fold 0"):
        seal_or_load_source_observation_evidence_authority(
            tmp_path / "missing.pt",
            heldout_fold=2,
            provenance=_provenance(),
            **_inputs(),
        )


@pytest.mark.parametrize("tamper", ["support", "magnitude", "provenance"])
def test_replay_fails_closed_on_material_difference(tmp_path: Path, tamper: str):
    path = tmp_path / "authority.pt"
    inputs = _inputs()
    seal_or_load_source_observation_evidence_authority(
        path,
        heldout_fold=0,
        provenance=_provenance(),
        **inputs,
    )
    candidate = dict(inputs)
    provenance = _provenance()
    if tamper == "support":
        value = inputs["positive_weight"].clone()
        value[1] = 0.1
        candidate["positive_weight"] = value
    elif tamper == "magnitude":
        value = inputs["raw_negative_mass"].clone()
        value[value != 0] *= 1.01
        candidate["raw_negative_mass"] = value
    else:
        provenance["scene_id"] = "different"
    with pytest.raises(ValueError):
        seal_or_load_source_observation_evidence_authority(
            path,
            heldout_fold=1,
            provenance=provenance,
            **candidate,
        )


def test_payload_tamper_is_detected(tmp_path: Path):
    path = tmp_path / "authority.pt"
    seal_or_load_source_observation_evidence_authority(
        path,
        heldout_fold=0,
        provenance=_provenance(),
        **_inputs(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["tensors"]["raw_positive_mass"][0] += 1.0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="changed"):
        seal_or_load_source_observation_evidence_authority(
            path,
            heldout_fold=0,
            provenance=_provenance(),
            **_inputs(),
        )
