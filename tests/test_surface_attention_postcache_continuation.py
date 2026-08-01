from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from radio_gs.scripts import surface_attention_postcache_continuation as authority


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    REPO_ROOT
    / "radio_gs/scripts/run_surface_attention_postcache_continuation.sh"
)

EXPECTED_PARENT_CACHE_STAGES = {
    "cache_control_c256_geometric_train_2",
    "cache_control_c256_geometric_train_3",
    "cache_control_c256_geometric_validation_0",
    "cache_control_c256_geometric_validation_1",
    "cache_context_c1024_geometric_train_0",
    "cache_context_c1024_geometric_train_1",
    "cache_context_c1024_geometric_train_2",
    "cache_context_c1024_geometric_train_3",
    "cache_context_c1024_geometric_validation_0",
    "cache_context_c1024_geometric_validation_1",
}


def _metadata() -> dict:
    return {
        "region_records": [{"region_id": "region-0"}],
        "scene_names": ["scene0001_00"],
        "split_role": "train",
        "split_file_sha256": "a" * 64,
        "teacher_target_source": "exact_cache_replay",
        "teacher_replay_cache": {
            "path": "/immutable/control.pt",
            "sha256": "b" * 64,
        },
        "teacher_replay_authority": {},
    }


def _sidecar(cache: Path, metadata: dict) -> dict:
    return {
        "output": str(cache.resolve()),
        "regions": len(metadata["region_records"]),
        "scenes": len(metadata["scene_names"]),
        "failed_scenes": {},
        "split_role": metadata["split_role"],
        "split_file_sha256": metadata["split_file_sha256"],
        "teacher_target_source": metadata["teacher_target_source"],
        "teacher_replay_cache": metadata["teacher_replay_cache"],
        "teacher_replay_authority": metadata["teacher_replay_authority"],
    }


def _write_sidecar(cache: Path, sidecar: dict) -> Path:
    path = cache.with_suffix(cache.suffix + ".json")
    path.write_text(json.dumps(sidecar) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("publish_empty_mapping", [False, True])
def test_conditional_sidecar_preserves_scene_intermediate_key_presence(
    tmp_path: Path,
    publish_empty_mapping: bool,
) -> None:
    cache = tmp_path / "cache.pt"
    cache.touch()
    metadata = _metadata()
    sidecar = _sidecar(cache, metadata)
    if publish_empty_mapping:
        metadata["scene_intermediate"] = {}
        sidecar["scene_intermediate"] = {}
    sidecar_path = _write_sidecar(cache, sidecar)

    record = authority._conditional_sidecar(cache, metadata, label="test cache")

    assert record["path"] == str(sidecar_path.resolve())


@pytest.mark.parametrize(
    "metadata_has_key,sidecar_has_key",
    [(False, True), (True, False)],
)
def test_conditional_sidecar_rejects_absent_empty_mapping_mismatch(
    tmp_path: Path,
    metadata_has_key: bool,
    sidecar_has_key: bool,
) -> None:
    cache = tmp_path / "cache.pt"
    cache.touch()
    metadata = _metadata()
    sidecar = _sidecar(cache, metadata)
    if metadata_has_key:
        metadata["scene_intermediate"] = {}
    if sidecar_has_key:
        sidecar["scene_intermediate"] = {}
    _write_sidecar(cache, sidecar)

    with pytest.raises(ValueError, match="sidecar differs"):
        authority._conditional_sidecar(cache, metadata, label="test cache")


def test_parent_cache_stage_contract_is_exactly_ten_stages() -> None:
    assert len(authority.EXPECTED_PARENT_CACHE_STAGES) == 10
    assert len(set(authority.EXPECTED_PARENT_CACHE_STAGES)) == 10
    assert set(authority.EXPECTED_PARENT_CACHE_STAGES) == EXPECTED_PARENT_CACHE_STAGES


def test_parent_inventory_accepts_exact_stages_independent_of_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = [
        {
            "stage": stage,
            "attempt_index": 1,
            "result": "completed",
            "receipt": {"path": f"/{stage}.json", "sha256": "c" * 64},
        }
        for stage in reversed(authority.EXPECTED_PARENT_CACHE_STAGES)
    ]
    inventory = {"attempts": attempts, "digest": "d" * 64}
    monkeypatch.setattr(
        authority,
        "audit_attempt_inventory",
        lambda **_kwargs: inventory,
    )

    observed = authority._validate_parent_inventory(
        tmp_path / "run_manifest.json",
        {
            "attempt_receipt_contract": {
                "root": str(tmp_path / "stage_attempts"),
                "log_root": str(tmp_path / "logs"),
            }
        },
    )

    assert observed is inventory


def test_parent_inventory_rejects_extra_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = [
        {"stage": stage, "attempt_index": 1, "result": "completed"}
        for stage in authority.EXPECTED_PARENT_CACHE_STAGES
    ]
    attempts.append(
        {"stage": "cache_unregistered", "attempt_index": 1, "result": "completed"}
    )
    monkeypatch.setattr(
        authority,
        "audit_attempt_inventory",
        lambda **_kwargs: {"attempts": attempts, "digest": "e" * 64},
    )

    with pytest.raises(ValueError, match="exact ten completed cache stages"):
        authority._validate_parent_inventory(
            tmp_path / "run_manifest.json",
            {
                "attempt_receipt_contract": {
                    "root": str(tmp_path / "stage_attempts"),
                    "log_root": str(tmp_path / "logs"),
                }
            },
        )


@pytest.mark.parametrize(
    "child",
    ("parent", "parent/child", "."),
)
def test_continuation_root_must_be_disjoint(
    tmp_path: Path,
    child: str,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    output = tmp_path if child == "." else tmp_path / child
    with pytest.raises(ValueError, match="must be disjoint"):
        authority._require_disjoint_roots(output, parent)


def test_continuation_root_accepts_sibling(tmp_path: Path) -> None:
    authority._require_disjoint_roots(tmp_path / "child", tmp_path / "parent")


def test_child_inventory_requires_exact_six_completed_readouts() -> None:
    attempts = [
        {"stage": stage, "attempt_index": 1, "result": "completed"}
        for stage in reversed(authority.EXPECTED_CHILD_READOUT_STAGES)
    ]
    authority._validate_child_inventory({"attempts": attempts})

    attempts.append(
        {"stage": "readout_extra", "attempt_index": 1, "result": "completed"}
    )
    with pytest.raises(ValueError, match="exact six completed readouts"):
        authority._validate_child_inventory({"attempts": attempts})


@pytest.mark.skipif(not RUNNER.is_file(), reason="continuation runner not added yet")
def test_postcache_runner_parses_and_stays_postcache_only() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    source = RUNNER.read_text(encoding="utf-8")
    assert "surface_attention_postcache_continuation.py" in source
    assert "verify-pairing" in source
    assert "train_surface_region_summary_readout.py" in source
    assert "finalize" in source
    assert "run_with_gpu_thermal_guard.sh" in source
    assert "surface_gpu1_lock_supervisor.py" in source
    assert "surface_region_run_guard.py" in source
    assert "build_scannet_surface_region_cache.py" not in source
    assert "cache_control_c256_geometric_" not in source
    assert "cache_context_c1024_geometric_" not in source
