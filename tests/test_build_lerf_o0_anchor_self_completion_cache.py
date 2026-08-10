from __future__ import annotations

import copy

import pytest
import torch

from radio_gs.scripts import build_lerf_o0_anchor_self_completion_cache as builder


def _inputs():
    state = {"path": "/tmp/state.pt", "sha256": "1" * 64}
    renderer = {"path": "/tmp/renderer.pt", "sha256": "2" * 64}
    accepted = {"path": "/tmp/accepted.pt", "sha256": "3" * 64}
    xyz = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    valid = torch.tensor([True, False, True, True])
    features = {
        "scene_id": "scene",
        "region_fingerprints": ["a", "b"],
        "input_authority": {
            "accepted_v2": accepted,
            "factorized_state": state,
        },
    }
    descriptor = {
        "scene_id": "scene",
        "region_fingerprints": ["a", "b"],
        "input_authority": {
            "target_accepted_v2": accepted,
            "factorized_primitive_state": state,
        },
    }
    o0 = {"metadata": {"renderer_geometry_checkpoint": renderer}}
    positive = {"xyz": xyz, "valid": valid}
    negative = {"xyz": xyz.clone(), "valid": valid.clone()}
    return state, renderer, xyz, valid, features, descriptor, o0, positive, negative


def _validate(values) -> None:
    state, renderer, xyz, valid, features, descriptor, o0, positive, negative = values
    builder._validate_region_primitive_lineage(
        scene_id="scene",
        features=features,
        descriptor=descriptor,
        state_record=state,
        state_xyz=xyz,
        renderer_record=renderer,
        renderer_xyz=xyz,
        o0=o0,
        positive=positive,
        negative=negative,
    )


def test_region_primitive_lineage_accepts_exact_chain() -> None:
    _validate(_inputs())


@pytest.mark.parametrize(
    "mutation",
    (
        "scene",
        "fingerprints",
        "accepted",
        "state_record",
        "state_xyz",
        "renderer_record",
        "renderer_xyz",
    ),
)
def test_region_primitive_lineage_fails_closed(mutation: str) -> None:
    values = list(copy.deepcopy(_inputs()))
    state, renderer, xyz, valid, features, descriptor, o0, positive, negative = values
    if mutation == "scene":
        descriptor["scene_id"] = "other"
    elif mutation == "fingerprints":
        descriptor["region_fingerprints"] = ["a", "c"]
    elif mutation == "accepted":
        descriptor["input_authority"]["target_accepted_v2"] = {
            "path": "/tmp/other.pt",
            "sha256": "4" * 64,
        }
    elif mutation == "state_record":
        features["input_authority"]["factorized_state"] = {
            "path": "/tmp/other.pt",
            "sha256": "4" * 64,
        }
    elif mutation == "state_xyz":
        positive["xyz"] = positive["xyz"].clone()
        positive["xyz"][0, 0] += 1
        negative["xyz"] = positive["xyz"].clone()
    elif mutation == "renderer_record":
        o0["metadata"]["renderer_geometry_checkpoint"] = {
            "path": "/tmp/other.pt",
            "sha256": "4" * 64,
        }
    elif mutation == "renderer_xyz":
        xyz[0, 0] += 1
    with pytest.raises(ValueError, match="region/primitive lineage"):
        _validate(values)


def test_score_change_summary_handles_legal_noop() -> None:
    assert builder._score_change_summary(torch.empty(0))["defined"] is False
    assert builder._score_change_summary(torch.empty(0))["minimum"] is None
    summary = builder._score_change_summary(torch.tensor([0.4, 0.5]))
    assert summary["defined"] is True
    assert summary["count"] == 2
