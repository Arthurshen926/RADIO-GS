from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

import radio_gs.scripts.bind_nvos_forward_beta_protocol_authority as authority_module
from radio_gs.scripts.bind_nvos_forward_beta_protocol_authority import (
    GENERAL_FREEZE_ID,
    GENERAL_FREEZE_SHA256,
    LUDVIG_GENERAL_TASK_SHA256,
    LUDVIG_PRIMARY_ROW_SHA256,
    LUDVIG_PROMPTABLE_ROW_SHA256,
    PARENT_FREEZE_SHA256,
    PARENT_NVOS_SUBCONTRACT_SHA256,
    PROMPTABLE_REGISTRY_SHA256,
    STRICT_PROTOCOL_ROW_SHA256,
    STRICT_SCORING_CONTRACT,
    AuthorityError,
    _load_exact_yaml,
    _validate_general_freeze_payload,
    _validate_promptable_registry_payload,
    build_authority,
    canonical_json_sha256,
    scoring_exactness,
    validate_authority_payload,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "paper/artifacts/canonical_mpr_v3_evaluation_freeze_20260716.yaml"
PROMPTABLE = ROOT / "paper/artifacts/promptable_nvs_protocol_registry.yaml"
GENERAL = ROOT / "paper/artifacts/evaluation_protocol_freeze_20260801.yaml"
METHOD_SHA = "a" * 64


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_canonical_authority_digests_are_pinned() -> None:
    parent = _load(PARENT)
    promptable = _load(PROMPTABLE)
    general = _load(GENERAL)

    assert canonical_json_sha256(
        parent["promptable_reconstruction"]["nvos"]
    ) == PARENT_NVOS_SUBCONTRACT_SHA256
    assert canonical_json_sha256(
        promptable["protocols"]["nvos_strict_unseen_v1"]
    ) == STRICT_PROTOCOL_ROW_SHA256
    assert canonical_json_sha256(
        promptable["protocols"][
            "ludvig_nvos_released_all_view_full8_exact_3seed_v1"
        ]
    ) == LUDVIG_PROMPTABLE_ROW_SHA256
    assert canonical_json_sha256(
        general["canonical_tasks"]["spatial_nvos_ludvig"]
    ) == LUDVIG_GENERAL_TASK_SHA256


def test_builds_double_positive_provenance_and_negative_fence() -> None:
    authority = build_authority(
        candidate_method_sha256=METHOD_SHA,
        scoring_contract=STRICT_SCORING_CONTRACT,
        repo_root=ROOT,
    )

    assert authority["strict_unseen_protocol_exact_match"] is True
    assert authority["strict_unseen_exact_match_blockers"] == []
    parent = authority["protocol_provenance"][
        "radio_gs_parent_evaluation_freeze"
    ]
    assert parent["file_sha256"] == PARENT_FREEZE_SHA256
    assert parent["parent_method_exact_match"] is False
    assert parent["candidate_method_contract_sha256"] == METHOD_SHA
    strict = authority["protocol_provenance"][
        "strict_unseen_benchmark_registry"
    ]
    assert strict["file_sha256"] == PROMPTABLE_REGISTRY_SHA256
    assert strict["protocol_row_canonical_json_sha256"] == (
        STRICT_PROTOCOL_ROW_SHA256
    )
    comparator = authority["external_comparator_provenance"]
    assert comparator["freeze_id"] == GENERAL_FREEZE_ID
    assert comparator["file_sha256"] == GENERAL_FREEZE_SHA256
    assert set(comparator["candidate_binding"].values()) == {None}
    assert comparator["excluded_from_candidate_authority"][
        "registry_row_canonical_json_sha256"
    ] == LUDVIG_PRIMARY_ROW_SHA256
    validate_authority_payload(authority)


@pytest.mark.parametrize(
    "scoring",
    [
        {
            "score_semantics": "beta_centered_posterior",
            "prediction_representation": "continuous_cosine_margin",
            "threshold": {"comparison": "greater_or_equal", "value": 0.0},
            "resize": "nearest",
        },
        {
            "score_semantics": "beta_foreground_posterior",
            "prediction_representation": "coverage_weighted_foreground_posterior",
            "threshold": {"comparison": "greater_or_equal", "value": 0.5},
            "resize": "cv2.INTER_LINEAR",
        },
    ],
)
def test_beta_scoring_contract_is_never_strict_exact(scoring: dict) -> None:
    exact, blockers = scoring_exactness(scoring)
    assert exact is False
    assert blockers
    if scoring["score_semantics"] == "beta_centered_posterior":
        assert blockers == ["score_semantics_differs"]


def test_candidate_method_sha_is_required_lowercase_64hex() -> None:
    for invalid in ("", "a" * 63, "A" * 64, "g" * 64):
        with pytest.raises(AuthorityError, match="64 lowercase hex"):
            build_authority(
                candidate_method_sha256=invalid,
                scoring_contract=STRICT_SCORING_CONTRACT,
                repo_root=ROOT,
            )


def test_parent_freeze_file_hash_drift_fails_closed(tmp_path: Path) -> None:
    drifted = tmp_path / PARENT.name
    drifted.write_bytes(PARENT.read_bytes() + b"\n")
    with pytest.raises(
        AuthorityError,
        match="parent evaluation freeze file SHA256 drifted",
    ):
        build_authority(
            candidate_method_sha256=METHOD_SHA,
            scoring_contract=STRICT_SCORING_CONTRACT,
            repo_root=ROOT,
            parent_freeze=drifted,
        )


def test_strict_registry_row_drift_fails_closed() -> None:
    payload = _load(PROMPTABLE)
    payload["protocols"]["nvos_strict_unseen_v1"]["evaluation"][
        "mask_resize"
    ] = "linear"
    with pytest.raises(AuthorityError, match="canonical row digest drifted"):
        _validate_promptable_registry_payload(payload)


def test_general_ludvig_task_drift_fails_closed() -> None:
    payload = _load(GENERAL)
    payload["canonical_tasks"]["spatial_nvos_ludvig"]["frozen_protocol"][
        "strict_unseen_claim"
    ] = True
    with pytest.raises(AuthorityError, match="canonical digest drifted"):
        _validate_general_freeze_payload(payload)


def test_ludvig_comparator_cannot_become_candidate_binding() -> None:
    authority = build_authority(
        candidate_method_sha256=METHOD_SHA,
        scoring_contract=STRICT_SCORING_CONTRACT,
        repo_root=ROOT,
    )
    forged = deepcopy(authority)
    forged["external_comparator_provenance"]["candidate_binding"] = {
        "canonical_task_id": "spatial_nvos_ludvig",
        "registry_row": "nvos_ludvig_released_all_view_full8_3seed_exact_20260731",
        "promptable_registry_row": (
            "ludvig_nvos_released_all_view_full8_exact_3seed_v1"
        ),
    }
    with pytest.raises(AuthorityError, match="cannot be bound"):
        validate_authority_payload(forged)


def test_parent_exact_match_cannot_be_forged_true() -> None:
    authority = build_authority(
        candidate_method_sha256=METHOD_SHA,
        scoring_contract=STRICT_SCORING_CONTRACT,
        repo_root=ROOT,
    )
    forged = json.loads(json.dumps(authority))
    forged["candidate"]["parent_method_exact_match"] = True
    with pytest.raises(AuthorityError, match="must remain false"):
        validate_authority_payload(forged)


def test_embedded_authority_hash_cannot_drift_with_recomputed_outer_digest() -> None:
    authority = build_authority(
        candidate_method_sha256=METHOD_SHA,
        scoring_contract=STRICT_SCORING_CONTRACT,
        repo_root=ROOT,
    )
    forged = deepcopy(authority)
    strict = forged["protocol_provenance"][
        "strict_unseen_benchmark_registry"
    ]
    strict["file_sha256"] = "b" * 64
    forged["protocol_provenance_sha256"] = canonical_json_sha256(
        forged["protocol_provenance"]
    )
    with pytest.raises(AuthorityError, match="strict protocol authority differs"):
        validate_authority_payload(forged)


def test_authority_reader_rejects_final_component_symlink(tmp_path: Path) -> None:
    link = tmp_path / "parent-freeze.yaml"
    link.symlink_to(PARENT)
    with pytest.raises(AuthorityError, match="refuses symlink"):
        _load_exact_yaml(
            link,
            expected_sha256=PARENT_FREEZE_SHA256,
            label="parent freeze test",
        )


def test_authority_reader_rejects_parent_directory_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    copied = real / PARENT.name
    copied.write_bytes(PARENT.read_bytes())
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(AuthorityError, match="unsafe directory component"):
        _load_exact_yaml(
            alias / PARENT.name,
            expected_sha256=PARENT_FREEZE_SHA256,
            label="parent freeze test",
        )


def test_builder_does_not_resolve_away_symlinked_repo_root(tmp_path: Path) -> None:
    alias = tmp_path / "repo-alias"
    alias.symlink_to(ROOT, target_is_directory=True)
    with pytest.raises(AuthorityError, match="unsafe directory component"):
        build_authority(
            candidate_method_sha256=METHOD_SHA,
            scoring_contract=STRICT_SCORING_CONTRACT,
            repo_root=alias,
        )


def test_authority_reader_rejects_file_drift_during_single_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = tmp_path / PARENT.name
    copied.write_bytes(PARENT.read_bytes())
    real_read = authority_module.os.read
    changed = False

    def read_then_change(descriptor: int, count: int) -> bytes:
        nonlocal changed
        block = real_read(descriptor, count)
        if block and not changed:
            changed = True
            with copied.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
        return block

    monkeypatch.setattr(authority_module.os, "read", read_then_change)
    with pytest.raises(AuthorityError, match="changed while it was being read"):
        _load_exact_yaml(
            copied,
            expected_sha256=PARENT_FREEZE_SHA256,
            label="parent freeze test",
        )
