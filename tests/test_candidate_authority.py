from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from radio_gs.candidate_authority import (
    EXPECTED_EVALUATION_CONTRACT_IDS,
    CandidateAuthorityError,
    build_candidate_authority,
    audit_candidate_authority,
    load_candidate_authority,
    reference_candidate_authority_inputs,
    validate_candidate_authority,
    write_candidate_authority,
)


def test_reference_candidate_authority_binds_the_five_contracts() -> None:
    bundle = build_candidate_authority(**reference_candidate_authority_inputs())

    assert bundle["schema_version"] == "radio_gs.candidate_authority.v1"
    assert len(bundle["candidate_id"]) == 64
    assert tuple(
        contract["contract_id"] for contract in bundle["evaluation_contracts"]
    ) == EXPECTED_EVALUATION_CONTRACT_IDS
    assert bundle["method_contract"]["field_schema"]["local_code_dimension"] == 512
    assert bundle["method_contract"]["field_schema"]["persistent_semantic_fields"] == 1


def test_candidate_identity_changes_when_a_bound_member_changes() -> None:
    inputs = reference_candidate_authority_inputs()
    original = build_candidate_authority(**inputs)

    changed = copy.deepcopy(inputs)
    changed["method_contract"]["implementation_identity"]["commit"] = "different"
    replacement = build_candidate_authority(**changed)

    assert replacement["candidate_id"] != original["candidate_id"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["evaluation_contracts"].__setitem__(
                0, {**value["evaluation_contracts"][0], "contract_id": "legacy-six-task"}
            ),
            "contract identity",
        ),
        (
            lambda value: value["evaluation_contracts"].reverse(),
            "frozen order",
        ),
        (
            lambda value: value["method_contract"]["field_schema"].update(
                {"second_semantic_tensor": True}
            ),
            "unexpected fields",
        ),
        (
            lambda value: value["method_contract"]["joint_mapping_objective"].update(
                {"identity": "lerf2d-objective"}
            ),
            "joint_mapping_objective identity",
        ),
        (
            lambda value: value["method_contract"]["mapping_checkpoint_rule"].update(
                {"identity": "scannet-checkpoint"}
            ),
            "mapping_checkpoint_rule identity",
        ),
        (
            lambda value: value["method_contract"]["modality_compilers"].update(
                {"by_benchmark": {"lerf2d": "different"}}
            ),
            "benchmark-conditioned",
        ),
        (
            lambda value: value["method_contract"]["output_domain_operators"].update(
                {"benchmark_renderer": {}}
            ),
            "benchmark-conditioned",
        ),
        (
            lambda value: value["evaluation_contracts"][2]["information_boundary"].update(
                {"query_captured_rgb": "forbidden"}
            ),
            "RGB-free",
        ),
        (
            lambda value: value["evaluation_contracts"][0]["information_boundary"].update(
                {"targets": "authorized"}
            ),
            "information boundary",
        ),
        (
            lambda value: value["evaluation_contracts"][0]["authorized_query_input"].update(
                {"private_siblings": []}
            ),
            "private sibling",
        ),
        (
            lambda value: value["method_contract"]["environment_identity"].update(
                {"dependency_lock_sha256": "+" + "0" * 63}
            ),
            "lowercase SHA-256",
        ),
    ],
)
def test_candidate_preflight_fails_closed_on_mutations(mutation, message: str) -> None:
    value = reference_candidate_authority_inputs()
    mutation(value)

    with pytest.raises(CandidateAuthorityError, match=message):
        build_candidate_authority(**value)


def test_candidate_bundle_has_a_canonical_content_address() -> None:
    bundle = build_candidate_authority(**reference_candidate_authority_inputs())

    encoded = bundle.canonical_json_bytes()
    decoded = json.loads(encoded)
    assert decoded == bundle.as_dict()
    assert encoded.endswith(b"\n")


def test_candidate_bundle_is_recursively_immutable() -> None:
    bundle = build_candidate_authority(**reference_candidate_authority_inputs())

    with pytest.raises(TypeError):
        bundle["method_contract"]["field_schema"]["local_code_dimension"] = 256
    with pytest.raises(TypeError):
        bundle["evaluation_contracts"][0]["contract_id"] = "legacy-six-task"


def test_candidate_authority_round_trips_through_immutable_artifact(tmp_path: Path) -> None:
    bundle = build_candidate_authority(**reference_candidate_authority_inputs())
    path = tmp_path / "authority.json"

    assert write_candidate_authority(path, bundle) == path
    loaded = load_candidate_authority(path)

    assert loaded.as_dict() == bundle.as_dict()
    assert audit_candidate_authority(path) == {
        "valid": True,
        "candidate_id": bundle["candidate_id"],
        "errors": [],
    }
    changed = reference_candidate_authority_inputs()
    changed["method_contract"]["implementation_identity"]["commit"] = "different"
    with pytest.raises(ValueError, match="already exists|differs"):
        write_candidate_authority(
            path,
            build_candidate_authority(**changed),
        )


def test_candidate_authority_rejects_content_drift_after_sealing(tmp_path: Path) -> None:
    bundle = build_candidate_authority(**reference_candidate_authority_inputs())
    path = tmp_path / "authority.json"
    write_candidate_authority(path, bundle)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["seed_policy"]["stochastic_seeds"] = [0, 1, 3]
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_candidate_authority(path)
    assert report["valid"] is False
    assert any("seed_policy" in error or "candidate_id" in error for error in report["errors"])
    with pytest.raises(CandidateAuthorityError, match="seed_policy"):
        validate_candidate_authority(payload)
