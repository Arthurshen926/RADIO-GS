from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from radio_gs.scripts import (
    build_region_comembership_v2_target_execution_authority as builder,
)


def _arguments(tmp_path: Path) -> argparse.Namespace:
    target_files = {}
    for name in (
        "accepted_v2",
        "typed_context",
        "support_graph",
        "factorized_state",
        "capability_descriptor",
    ):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode())
        target_files[name] = str(path)
    return argparse.Namespace(
        scene_id="target",
        four_plus_two_result=str(tmp_path / "result.json"),
        expected_four_plus_two_result_sha256="0" * 64,
        promoted_checkpoint=str(tmp_path / "checkpoint.pt"),
        expected_promoted_checkpoint_sha256="0" * 64,
        target_feature_output=str((tmp_path / "features.pt").resolve()),
        target_inference_output=str((tmp_path / "inference.pt").resolve()),
        output=str(tmp_path / "execution.json"),
        **target_files,
    )


def test_source_failure_occurs_before_any_target_input_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _arguments(tmp_path)
    opened: list[str] = []

    def fail_source(*_args, **_kwargs):
        raise ValueError("source gate failed")

    def record_target(path):
        opened.append(str(path))
        return {"path": str(path), "sha256": "1" * 64}

    monkeypatch.setattr(builder, "load_json_object", fail_source)
    monkeypatch.setattr(builder, "file_record", record_target)
    with pytest.raises(ValueError, match="source gate failed"):
        builder.build(args)
    assert opened == []


def test_existing_outputs_fail_before_source_or_target_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _arguments(tmp_path)
    Path(args.output).write_text("occupied")
    touched = False

    def touch(*_args, **_kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("must not be called")

    monkeypatch.setattr(builder, "load_json_object", touch)
    with pytest.raises(FileExistsError, match="authority exists"):
        builder.build(args)
    assert touched is False


def test_target_outputs_must_be_absolute_and_new(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute canonical"):
        builder._canonical_new_output("relative.pt", label="test output")
    existing = tmp_path / "existing.pt"
    existing.write_bytes(b"x")
    with pytest.raises(FileExistsError, match="already exists"):
        builder._canonical_new_output(str(existing), label="test output")
