from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path

import pytest

from radio_gs.scripts import surface_text_response_distill_authority as authority
from radio_gs.scripts import surface_attention_pooling_screen as pooling


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "radio_gs/scripts/run_surface_region_text_response_distill.sh"
CURRENT_POSTCACHE_ROOT = Path(
    "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/"
    "surface_c1024_attention_postcache_v1_gpu1only_src1b85cfdaf7b5"
)
JOINT = "joint_attention_v1"
SEPARATE = "core_context_separate_attention_v1"
CANDIDATE = "context_c1024_geometric"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _record(path: Path) -> dict[str, str]:
    return authority.file_record(path)


def _postcache_fixture(tmp_path: Path) -> dict[str, object]:
    surface = tmp_path / "attention_postcache"
    surface.mkdir()
    parent = tmp_path / "immutable_parent"
    parent.mkdir()

    run_manifest = surface / "run_manifest.json"
    parent_manifest = parent / "run_manifest.json"
    _write_json(parent_manifest, {"schema_version": 1, "screen": "parent-test"})

    train: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    sidecars: list[dict[str, str]] = []
    pairing_rows: list[dict[str, object]] = []
    for role, count in (("train", 4), ("validation", 2)):
        for shard in range(count):
            cache = parent / "caches" / CANDIDATE / f"{role}_shard{shard}.pt"
            sidecar = cache.with_suffix(cache.suffix + ".json")
            control = parent / "controls" / f"{role}_shard{shard}.pt"
            control_sidecar = control.with_suffix(control.suffix + ".json")
            cache.parent.mkdir(parents=True, exist_ok=True)
            control.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(f"c1024-{role}-{shard}".encode())
            _write_json(sidecar, {"role": role, "shard": shard})
            control.write_bytes(f"control-{role}-{shard}".encode())
            _write_json(control_sidecar, {"role": role, "shard": shard})
            cache_record = _record(cache)
            sidecar_record = _record(sidecar)
            (train if role == "train" else validation).append(cache_record)
            sidecars.append(sidecar_record)
            pairing_rows.append(
                {
                    "role": role,
                    "shard": shard,
                    "regions": 96,
                    "teacher_target_protocol_sha256": "a" * 64,
                    "control": _record(control),
                    "control_sidecar": _record(control_sidecar),
                    "c1024": cache_record,
                    "c1024_sidecar": sidecar_record,
                }
            )

    selection_contract = {
        "baseline": JOINT,
        "candidate": SEPARATE,
        "descriptor_components": [
            "mean_descriptor_cosine",
            "all_view_descriptor_cosine",
        ],
        "maximum_descriptor_component_drop": 0.002,
        "minimum_mean_score_gain": 0.001,
        "minimum_seed_wins": 2,
        "uses_benchmark_queries": False,
    }
    _write_json(
        run_manifest,
        {
            "schema_version": 1,
            "screen": "surface-c1024-attention-postcache-continuation-v1",
            "continuation_contract": {
                "mode": "post_cache_only_no_parent_mutation_v1",
                "cache_writes_forbidden": True,
                "parent_run_manifest": _record(parent_manifest),
            },
            "cache_bundle": pairing_rows,
            "runtime_closure": {"digest": "c" * 64},
            "selection_contract": selection_contract,
        },
    )

    pairing_path = surface / "cache_pairing.json"
    pairing = {
        "schema_version": 1,
        "artifact_type": "surface_c1024_exact_teacher_pairing",
        "status": "single_c1024_cache_exact_teacher_replay_verified",
        "run_manifest": _record(run_manifest),
        "parent_run_manifest": _record(parent_manifest),
        "rows": pairing_rows,
        "legacy_sidecar_rule": (
            "scene_intermediate_key_absent_iff_frozen_builder_metadata_omits_key"
        ),
        "benchmark_queries_opened": False,
        "benchmark_masks_opened": False,
    }
    _write_json(pairing_path, pairing)

    variants: dict[str, dict] = {}
    joint_checkpoints: list[dict[str, str]] = []
    for variant in (JOINT, SEPARATE):
        seeds = []
        score = 0.9 if variant == JOINT else 0.899
        validation_metrics = {
            "summary_token_cosine": 0.8,
            "mean_descriptor_cosine": score,
            "all_view_descriptor_cosine": score,
        }
        for seed in range(3):
            checkpoint = surface / "readouts" / f"{variant}_seed{seed}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{variant}-seed-{seed}".encode())
            checkpoint_record = _record(checkpoint)
            if variant == JOINT:
                joint_checkpoints.append(checkpoint_record)
            checkpoint_report = {
                "checkpoint_sha256": checkpoint_record["sha256"],
                "best_epoch": 1,
                "best_selection_score": score,
                "validation": validation_metrics,
            }
            _write_json(
                checkpoint.with_suffix(checkpoint.suffix + ".json"),
                checkpoint_report,
            )
            seeds.append(
                {
                    "seed": seed,
                    "checkpoint": checkpoint_record,
                    "best_epoch": 1,
                    "best_selection_score": score,
                    "validation": validation_metrics,
                }
            )
        variants[variant] = {
            "seeds": seeds,
            "mean_selection_score": score,
            "mean_validation": validation_metrics,
        }
    variants[SEPARATE].update(
        pooling.promotion_decision(variants, selection_contract)
    )
    joint_controls = [
        {
            "seed": row["seed"],
            "checkpoint": row["checkpoint"],
            "best_epoch": row["best_epoch"],
            "best_selection_score": row["best_selection_score"],
            "validation": row["validation"],
        }
        for row in variants[JOINT]["seeds"]
    ]

    receipt_records = []
    inventory_attempts = []
    for variant in (JOINT, SEPARATE):
        for seed in range(3):
            stage = f"readout_{variant}_seed{seed}"
            receipt = surface / "stage_attempts" / stage / "attempt_000001.json"
            checkpoint_record = variants[variant]["seeds"][seed]["checkpoint"]
            _write_json(
                receipt,
                {
                    "stage": stage,
                    "attempt_index": 1,
                    "result": "completed",
                    "command_status": 0,
                    "run_manifest": _record(run_manifest),
                    "terminal": checkpoint_record,
                },
            )
            receipt_record = _record(receipt)
            receipt_records.append(receipt_record)
            inventory_attempts.append(
                {
                    "stage": stage,
                    "attempt_index": 1,
                    "result": "completed",
                    "command_status": 0,
                    "run_manifest": _record(run_manifest),
                    "receipt": receipt_record,
                }
            )

    screen_path = surface / "attention_pooling_screen.json"
    screen = {
        "schema_version": 1,
        "artifact_type": "surface_c1024_attention_pooling_postcache_continuation",
        "selection_status": "joint_attention_retained",
        "selected_variant": JOINT,
        "promotion_gate_passed": False,
        "run_manifest": _record(run_manifest),
        "parent_run_manifest": _record(parent_manifest),
        "cache_pairing_report": _record(pairing_path),
        "child_attempt_inventory_digest": "b" * 64,
        "child_attempt_receipts": sorted(
            receipt_records, key=lambda record: record["path"]
        ),
        "variants": variants,
        "benchmark_queries_opened": False,
        "benchmark_masks_opened": False,
        "next_gate": "freeze winning readout before text-response benchmark",
    }
    _write_json(screen_path, screen)

    closure_path = surface / "runtime_closure_final.json"
    closure = {
        "schema_version": 1,
        "artifact_type": "surface_region_runtime_closure_audit",
        "status": "runtime_closure_verified",
        "phase": "final_before_completion",
        "run_manifest": str(run_manifest.resolve()),
        "run_manifest_sha256": _record(run_manifest)["sha256"],
        "runtime_closure_digest": "c" * 64,
        "radio_checkpoint_sha256": "d" * 64,
        "full_checkpoint_rehashed": True,
        "attempt_inventory": {
            "artifact_type": "surface-region-stage-attempt-inventory-v1",
            "schema_version": 1,
            "run_manifest": _record(run_manifest),
            "attempt_root": str((surface / "stage_attempts").resolve()),
            "log_root": str((surface / "logs").resolve()),
            "attempts": inventory_attempts,
            "digest": "b" * 64,
        },
    }
    _write_json(closure_path, closure)
    completion_path = surface / "screen.complete"
    completion_path.write_text("2026-08-01T16:25:31+08:00\n", encoding="utf-8")
    return {
        "surface": surface,
        "train": train,
        "validation": validation,
        "sidecars": sidecars,
        "joint_checkpoints": joint_checkpoints,
        "joint_controls": joint_controls,
        "pairing_path": pairing_path,
        "screen_path": screen_path,
        "closure_path": closure_path,
        "completion_path": completion_path,
        "run_manifest": run_manifest,
    }


def _bind(fixture: dict[str, object]) -> dict[str, object]:
    return authority._surface_binding(
        surface_root=fixture["surface"],
        candidate=CANDIDATE,
        train=fixture["train"],
        validation=fixture["validation"],
    )


def test_surface_binding_accepts_joint_attention_postcache(
    tmp_path: Path,
) -> None:
    fixture = _postcache_fixture(tmp_path)

    binding = _bind(fixture)

    assert binding == {
        "binding_mode": "attention_postcache_joint_v1",
        "run_manifest": _record(fixture["run_manifest"]),
        "cache_pairing": _record(fixture["pairing_path"]),
        "attention_pooling_screen": _record(fixture["screen_path"]),
        "screen_completion": _record(fixture["completion_path"]),
        "runtime_closure_final": _record(fixture["closure_path"]),
        "selected_variant": JOINT,
        "selected_readouts": fixture["joint_controls"],
        "selected_cache_sidecars": fixture["sidecars"],
    }


@pytest.mark.skipif(
    not (CURRENT_POSTCACHE_ROOT / "screen.complete").is_file(),
    reason="formal post-cache continuation is not mounted",
)
def test_surface_binding_accepts_current_formal_postcache_root() -> None:
    pairing = _load_json(CURRENT_POSTCACHE_ROOT / "cache_pairing.json")
    rows = pairing["rows"]
    train = [row["c1024"] for row in rows if row["role"] == "train"]
    validation = [
        row["c1024"] for row in rows if row["role"] == "validation"
    ]

    binding = authority._surface_binding(
        surface_root=CURRENT_POSTCACHE_ROOT,
        candidate=CANDIDATE,
        train=train,
        validation=validation,
    )

    assert binding["binding_mode"] == "attention_postcache_joint_v1"
    assert binding["selected_variant"] == JOINT
    assert len(binding["selected_readouts"]) == 3
    assert len(binding["selected_cache_sidecars"]) == 6


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selected_variant", SEPARATE),
        (
            "selection_status",
            "separate_attention_promoted_benchmark_gate_still_closed",
        ),
    ],
)
def test_surface_binding_rejects_separate_attention_as_retained_selection(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    fixture = _postcache_fixture(tmp_path)
    screen_path = fixture["screen_path"]
    screen = _load_json(screen_path)
    screen[field] = value
    _write_json(screen_path, screen)

    with pytest.raises(ValueError):
        _bind(fixture)


@pytest.mark.parametrize(
    ("artifact", "field"),
    [
        ("screen_path", "benchmark_queries_opened"),
        ("screen_path", "benchmark_masks_opened"),
        ("pairing_path", "benchmark_queries_opened"),
        ("pairing_path", "benchmark_masks_opened"),
    ],
)
def test_surface_binding_rejects_open_query_or_mask_evidence(
    tmp_path: Path,
    artifact: str,
    field: str,
) -> None:
    fixture = _postcache_fixture(tmp_path)
    path = fixture[artifact]
    payload = _load_json(path)
    payload[field] = True
    _write_json(path, payload)

    with pytest.raises(ValueError):
        _bind(fixture)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_surface_binding_requires_exact_joint_seed_set(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _postcache_fixture(tmp_path)
    screen_path = fixture["screen_path"]
    screen = _load_json(screen_path)
    seeds = screen["variants"][JOINT]["seeds"]
    if mutation == "missing":
        seeds.pop()
    else:
        extra = copy.deepcopy(seeds[-1])
        extra["seed"] = 3
        seeds.append(extra)
    _write_json(screen_path, screen)

    with pytest.raises(ValueError):
        _bind(fixture)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_surface_binding_requires_exact_six_c1024_cache_records(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _postcache_fixture(tmp_path)
    pairing_path = fixture["pairing_path"]
    pairing = _load_json(pairing_path)
    if mutation == "missing":
        pairing["rows"].pop()
    else:
        pairing["rows"].append(copy.deepcopy(pairing["rows"][-1]))
    _write_json(pairing_path, pairing)

    with pytest.raises(ValueError):
        _bind(fixture)


def test_surface_binding_rejects_caches_that_differ_from_consumer_inputs(
    tmp_path: Path,
) -> None:
    fixture = _postcache_fixture(tmp_path)
    fixture["train"] = list(fixture["train"])[1:]

    with pytest.raises(ValueError):
        _bind(fixture)


@pytest.mark.parametrize("artifact", ["cache", "checkpoint"])
def test_surface_binding_rejects_selected_artifact_sha_drift(
    tmp_path: Path,
    artifact: str,
) -> None:
    fixture = _postcache_fixture(tmp_path)
    if artifact == "cache":
        path = Path(fixture["train"][0]["path"])
    else:
        path = Path(fixture["joint_checkpoints"][0]["path"])
    path.write_bytes(path.read_bytes() + b"-tampered")

    with pytest.raises(ValueError):
        _bind(fixture)


def test_surface_binding_rejects_ambiguous_legacy_and_attention_layouts(
    tmp_path: Path,
) -> None:
    fixture = _postcache_fixture(tmp_path)
    _write_json(
        fixture["surface"] / "query_free_promotion_bundle.json",
        {"selected_candidate": CANDIDATE},
    )

    with pytest.raises(ValueError):
        _bind(fixture)


def test_text_response_runner_freezes_gpu1_only_attention_consumer() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    source = RUNNER.read_text(encoding="utf-8")

    assert 'GPU="${GPU:-1}"' in source
    assert 'GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-78}"' in source
    assert 'GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-65}"' in source
    assert 'GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-3}"' in source
    assert 'GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-75}"' in source
    assert 'GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-70}"' in source
    assert 'GPU_PEER_INDEX=""' in source
    assert 'GPU_MAX_POWER_LIMIT_W="${GPU_MAX_POWER_LIMIT_W:-300.5}"' in source
    for name in (
        "attention_pooling_screen.json",
        "runtime_closure_final.json",
        "cache_pairing.json",
        "screen.complete",
    ):
        assert f'$SURFACE_ROOT/{name}' in source
    assert "bind_surface_control" in source
    assert "--surface-control-checkpoint" in source
    assert "--surface-control-checkpoint-sha256" in source
    assert "query_free_promotion_bundle.json" in source

    compute_queries = re.findall(
        r"nvidia-smi(?P<arguments>[^\n]*--query-compute-apps[^\n]*)",
        source,
    )
    assert all(re.search(r'(?:^|\s)-i\s+"?\$GPU"?(?:\s|$)', row) for row in compute_queries)
