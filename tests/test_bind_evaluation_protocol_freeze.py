from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from radio_gs.scripts.bind_evaluation_protocol_freeze import (
    ARTIFACT_TYPE,
    EXTERNAL_SCOPE,
    UNOPENED_SCOPE,
    BindingError,
    build_binding,
    main,
    verify_binding_receipt,
    write_binding_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "paper/artifacts/evaluation_protocol_freeze_20260801.yaml"
TASK_ID = "spatial_nvos_ludvig"
REGISTRY_ROW = "nvos_ludvig_released_all_view_full8_3seed_exact_20260731"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_binding_matches_exact_frozen_task_and_row() -> None:
    binding = build_binding(
        FREEZE,
        scope=EXTERNAL_SCOPE,
        canonical_task_id=TASK_ID,
        registry_row=REGISTRY_ROW,
        repo_root=ROOT,
    )

    assert binding["artifact_type"] == ARTIFACT_TYPE
    assert binding["scope"] == EXTERNAL_SCOPE
    assert binding["freeze"] == {
        "path": str(FREEZE.resolve()),
        "sha256": "af91f0861d3a15354063579e78f64898801c41f2543d1cf9b352a0a123820916",
        "freeze_id": "evaluation_protocols_20260801_v1",
    }
    assert binding["freeze"]["sha256"] == _sha256(FREEZE)
    assert binding["task"] == {
        "canonical_task_id": TASK_ID,
        "registry_row": REGISTRY_ROW,
    }
    assert binding["validation"]["authoritative_artifact_hashes_verified"] is True


@pytest.mark.parametrize(
    ("task_id", "registry_row", "message"),
    [
        (TASK_ID, None, "requires canonical_task_id and registry_row"),
        ("not_frozen", REGISTRY_ROW, "not selected by the freeze"),
        (TASK_ID, "wrong_row", "does not match"),
    ],
)
def test_external_binding_fails_closed_on_missing_or_mismatched_selection(
    task_id: str,
    registry_row: str | None,
    message: str,
) -> None:
    with pytest.raises(BindingError, match=message):
        build_binding(
            FREEZE,
            scope=EXTERNAL_SCOPE,
            canonical_task_id=task_id,
            registry_row=registry_row,
            repo_root=ROOT,
        )


def test_internal_binding_explicitly_keeps_external_benchmarks_unopened() -> None:
    binding = build_binding(
        FREEZE,
        scope=UNOPENED_SCOPE,
        repo_root=ROOT,
    )
    assert binding["scope"] == "external_benchmarks_unopened"
    assert binding["task"] is None

    with pytest.raises(BindingError, match="requires task to be null"):
        build_binding(
            FREEZE,
            scope=UNOPENED_SCOPE,
            canonical_task_id=TASK_ID,
            registry_row=REGISTRY_ROW,
            repo_root=ROOT,
        )


def test_receipt_is_atomic_and_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "freeze_binding.json"
    binding = build_binding(
        FREEZE,
        scope=UNOPENED_SCOPE,
        repo_root=ROOT,
    )
    assert write_binding_receipt(output, binding) == output
    assert json.loads(output.read_text(encoding="utf-8")) == binding

    with pytest.raises(BindingError, match="already exists"):
        write_binding_receipt(output, {"forged": True})
    assert json.loads(output.read_text(encoding="utf-8")) == binding
    assert not list(output.parent.glob("*.tmp"))


def test_cli_writes_unopened_manifest_fragment(tmp_path: Path) -> None:
    output = tmp_path / "binding.json"
    assert (
        main(
            [
                "--freeze",
                str(FREEZE),
                "--repo-root",
                str(ROOT),
                "--scope",
                UNOPENED_SCOPE,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scope"] == UNOPENED_SCOPE
    assert payload["task"] is None


def test_existing_receipt_resume_requires_exact_binding(tmp_path: Path) -> None:
    output = tmp_path / "binding.json"
    binding = build_binding(FREEZE, scope=UNOPENED_SCOPE, repo_root=ROOT)
    write_binding_receipt(output, binding)
    assert verify_binding_receipt(output, binding) == output

    forged = {**binding, "scope": EXTERNAL_SCOPE}
    with pytest.raises(BindingError, match="differs"):
        verify_binding_receipt(output, forged)

    alias = tmp_path / "alias.json"
    alias.symlink_to(output)
    with pytest.raises(BindingError, match="cannot be opened safely"):
        verify_binding_receipt(alias, binding)


def test_cli_verifies_exact_existing_receipt(tmp_path: Path) -> None:
    output = tmp_path / "binding.json"
    binding = build_binding(FREEZE, scope=UNOPENED_SCOPE, repo_root=ROOT)
    write_binding_receipt(output, binding)
    assert (
        main(
            [
                "--freeze",
                str(FREEZE),
                "--repo-root",
                str(ROOT),
                "--scope",
                UNOPENED_SCOPE,
                "--output",
                str(output),
                "--verify-existing",
            ]
        )
        == 0
    )
