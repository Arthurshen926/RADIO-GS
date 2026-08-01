from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from radio_gs.interfaces.surface_region_summary import (
    JOINT_CONTEXT_POOLING,
    SEPARATE_CONTEXT_POOLING,
)
from radio_gs.scripts import surface_attention_pooling_screen as authority


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "radio_gs/scripts/run_surface_attention_pooling_screen.sh"
BUILDER = REPO_ROOT / "radio_gs/scripts/build_scannet_surface_region_cache.py"
LEGACY_ROOT = (
    REPO_ROOT
    / "output/optimization_20260731/"
    "surface_fixed_teacher_replay_v2_gpu1_p8_hard75"
)


def _variant(
    scores: tuple[float, float, float],
    *,
    summary: float,
    mean_descriptor: float,
    all_view: float,
) -> dict:
    return {
        "seeds": [
            {"seed": seed, "best_selection_score": score}
            for seed, score in zip(authority.SEEDS, scores)
        ],
        "mean_selection_score": sum(scores) / len(scores),
        "mean_validation": {
            "summary_token_cosine": summary,
            "mean_descriptor_cosine": mean_descriptor,
            "all_view_descriptor_cosine": all_view,
        },
    }


def _gate() -> dict:
    return {
        "minimum_mean_score_gain": 0.001,
        "minimum_seed_wins": 2,
        "maximum_descriptor_component_drop": 0.002,
    }


def test_runner_is_an_isolated_same_cache_attention_screen() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "run_surface_region_context_recovery_screen.sh" not in source
    assert 'READOUT_SEEDS="${READOUT_SEEDS:-0,1,2}"' in source
    assert 'if [[ "$READOUT_SEEDS" != "0,1,2" ]]' in source
    assert "for pooling_mode in joint_attention_v1 core_context_separate_attention_v1" in source
    assert '--context-pooling-mode "$pooling_mode"' in source
    assert "context_c1024_geometric/train_shard*.pt" in source
    assert "context_c1024_geometric/validation_shard*.pt" in source
    assert "context_c1024_uniform" not in source
    assert "core_c1024_geometric" not in source
    assert "--token-candidate-limit 1024" in source
    assert "--region-reliability-mode geometric_mean_observation_agreement" in source
    assert "--teacher-replay-cache \"$replay\"" in source
    assert "run_authority legacy-replay-authority" in source
    assert 'replay_authority_output="$(' in source
    assert 'realpath -e -- "$replay_authority"' in source
    assert '"${replay_authority_lines[0]}" != "$replay_authority_canonical"' in source
    assert '^[0-9a-f]{64}$' in source
    assert "--teacher-replay-authority \"$replay_authority\"" in source
    assert (
        "--teacher-replay-authority-sha256 \"$replay_authority_sha\""
        in source
    )
    builder = BUILDER.read_text(encoding="utf-8")
    assert "exact_historical_cache_fixed_teacher_replay_only" in builder
    assert "teacher_replay_authority" in builder


def test_runner_reuses_only_proven_external_controls_and_builds_missing() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "validate_external_only >/dev/null" in source
    assert 'SURFACE_ALLOW_VALIDATED_EXTERNAL_CONTROL_REUSE="${SURFACE_ALLOW_VALIDATED_EXTERNAL_CONTROL_REUSE:-0}"' in source
    assert "full run is fail-closed: explicitly set" in source
    assert 'if [[ "$role" == "train" && "$shard" -lt 2 ]]' in source
    assert "build_local_control train 2" in source
    assert "build_local_control train 3" in source
    assert "build_local_control validation 0" in source
    assert "build_local_control validation 1" in source
    assert "build_local_control train 0" not in source
    assert "build_local_control train 1" not in source
    assert authority.LEGACY_CONTROL_SHA256 == {
        0: "02cfa45af46cf8274c17ccde28e8953fb4b659527652525340703a241cad22bc",
        1: "dfd2c0857aca2495b867da393b4cda298554670e313334392bdc030876d84460",
    }
    assert authority.LEGACY_SIDECAR_SHA256 == {
        0: "37f19d5778b59e9efb32af97d0c9eeec3daf7b2ebf9f88c9886886a9aaf33e0c",
        1: "4b3dff0f57d13b1a387b13246cff85ad0dbb1de9520d176db4d567112b1047d2",
    }
    assert authority.LEGACY_BUILDER_SHA256 == (
        "182408a3f16dcd8a50b0190157c885de81d35a87b59a1f5262b7ed6d81ab8d63"
    )


def test_balanced_p6_defaults_and_override_provenance_are_explicit() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    expected = (
        'GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-78}"',
        'GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-65}"',
        'GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-3}"',
        'GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-75}"',
        'GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-70}"',
        'GPU_PEER_INDEX=""',
        "GPU_PEER_PAUSE_TEMP_C=0",
        "GPU_PEER_RESUME_TEMP_C=0",
        "GPU_PEER_MAX_POWER_W=0",
        'GPU_OWNER_PID_NAMESPACE_MODE="exclusive-singleton-after-clear-v1"',
        'RADIO_THERMAL_PACING_SECONDS_PER_IMAGE="${RADIO_THERMAL_PACING_SECONDS_PER_IMAGE:-6.0}"',
        'SURFACE_CANARY_MAX_TEMP_C="${SURFACE_CANARY_MAX_TEMP_C:-74}"',
    )
    for fragment in expected:
        assert fragment in source
    assert "GPU_MAX_TEMP_C_SOURCE=\"$(value_source GPU_MAX_TEMP_C)\"" in source
    assert "RADIO_PACING_SOURCE=\"$(value_source RADIO_THERMAL_PACING_SECONDS_PER_IMAGE)\"" in source
    assert '"override_provenance"' in Path(authority.__file__).read_text(
        encoding="utf-8"
    )
    assert "p8 run stayed <=71C" in source
    assert "p4 observations stayed" in source and "<=73C" in source
    assert "requires physical peer GPU0" not in source
    assert '"peer_gpu": None if not peer else int(peer)' in source
    assert 'nvidia-smi -i "$GPU" --query-compute-apps' in source
    assert 'owner_audit="$attempt_dir/attempt_${attempt_tag}.owner_audit.csv"' in source
    assert '"owner_audit": file_record(owner_audit)' in source


def test_intermediate_fastpath_is_explicit_and_never_silent() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    builder = BUILDER.read_text(encoding="utf-8")
    assert 'SURFACE_INTERMEDIATE_FASTPATH="${SURFACE_INTERMEDIATE_FASTPATH:-required_local_shards}"' in source
    for option in (
        "--scene-intermediate-output-root",
        "--scene-intermediate-manifest",
        "--scene-intermediate-manifest-sha256",
    ):
        assert option in source
        assert option in builder
    assert "required Surface intermediate fastpath is unavailable" in source
    assert "run_authority intermediate-binding" in source
    assert '--control-sidecar "${replay}.json"' in source
    assert "sha256sum \"$intermediate_manifest\"" not in source
    assert 'SURFACE_INTERMEDIATE_FASTPATH" == "required_local_shards"' in source
    assert "legacy train0/1 use the full builder" in Path(authority.__file__).read_text(
        encoding="utf-8"
    )


def test_intermediate_binding_comes_from_control_sidecar(
    tmp_path: Path,
) -> None:
    root = tmp_path / "intermediate"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text('{"authority":"test"}\n', encoding="utf-8")
    record = {
        "path": str(manifest.resolve()),
        "sha256": authority.sha256_file(manifest),
    }
    sidecar = tmp_path / "control.pt.json"
    sidecar.write_text(
        json.dumps(
            {
                "scene_intermediate": {
                    "mode": "fresh_publish",
                    "root": str(root.resolve()),
                    "manifest": record,
                    "scene_records": [{"scene": "scene0001_00"}],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    binding = authority.intermediate_binding(sidecar, root)
    assert binding["manifest"] == record
    assert binding["root"] == str(root.resolve())


@pytest.mark.parametrize("with_intermediate", [False, True])
def test_current_cache_sidecar_mirrors_optional_intermediate_schema(
    tmp_path: Path,
    with_intermediate: bool,
) -> None:
    cache = tmp_path / "cache.pt"
    cache.touch()
    metadata = {
        "region_records": [{"region_id": "region-0"}],
        "scene_names": ["scene0001_00"],
        "split_role": "train",
        "split_file_sha256": "a" * 64,
        "teacher_target_source": "exact_cache_replay",
        "teacher_replay_cache": {"path": "/cache.pt", "sha256": "b" * 64},
        "teacher_replay_authority": {},
    }
    sidecar = {
        "output": str(cache.resolve()),
        "regions": 1,
        "scenes": 1,
        "failed_scenes": {},
        "split_role": "train",
        "split_file_sha256": "a" * 64,
        "teacher_target_source": "exact_cache_replay",
        "teacher_replay_cache": {"path": "/cache.pt", "sha256": "b" * 64},
        "teacher_replay_authority": {},
    }
    if with_intermediate:
        provenance = {"mode": "exact_replay", "manifest": {"sha256": "c" * 64}}
        metadata["scene_intermediate"] = provenance
        sidecar["scene_intermediate"] = provenance
    cache.with_suffix(".pt.json").write_text(
        json.dumps(sidecar) + "\n", encoding="utf-8"
    )

    record = authority._validate_current_cache_sidecar(
        cache, metadata, label="test cache"
    )
    assert record["path"] == str(cache.with_suffix(".pt.json").resolve())


def test_current_cache_sidecar_rejects_unpublished_empty_intermediate(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache.pt"
    cache.touch()
    metadata = {
        "region_records": [{"region_id": "region-0"}],
        "scene_names": ["scene0001_00"],
        "split_role": "train",
        "split_file_sha256": "a" * 64,
        "teacher_target_source": "exact_cache_replay",
        "teacher_replay_cache": {},
        "teacher_replay_authority": {},
    }
    sidecar = {
        "output": str(cache.resolve()),
        "regions": 1,
        "scenes": 1,
        "failed_scenes": {},
        "split_role": "train",
        "split_file_sha256": "a" * 64,
        "teacher_target_source": "exact_cache_replay",
        "teacher_replay_cache": {},
        "teacher_replay_authority": {},
        "scene_intermediate": {},
    }
    cache.with_suffix(".pt.json").write_text(
        json.dumps(sidecar) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="sidecar differs"):
        authority._validate_current_cache_sidecar(
            cache, metadata, label="test cache"
        )


def test_promotion_gate_passes_at_frozen_boundaries_and_ignores_token_diagnostic() -> None:
    rows = {
        JOINT_CONTEXT_POOLING: _variant(
            (0.50, 0.51, 0.52),
            summary=0.95,
            mean_descriptor=0.80,
            all_view=0.82,
        ),
        SEPARATE_CONTEXT_POOLING: _variant(
            (0.502, 0.512, 0.519),
            # Summary token cosine is diagnostic, not a descriptor component.
            summary=0.10,
            mean_descriptor=0.798,
            all_view=0.818,
        ),
    }
    decision = authority.promotion_decision(rows, _gate())
    assert decision["mean_score_gain_over_joint"] == pytest.approx(0.001)
    assert decision["seed_wins_over_joint"] == 2
    assert decision["descriptor_component_drops_from_joint"] == pytest.approx(
        {"mean_descriptor_cosine": 0.002, "all_view_descriptor_cosine": 0.002}
    )
    assert decision["eligible_for_query_free_promotion"] is True


@pytest.mark.parametrize(
    "candidate_scores,mean_descriptor,all_view",
    [
        ((0.501, 0.509, 0.519), 0.80, 0.82),  # only one paired seed win
        ((0.502, 0.512, 0.519), 0.7979, 0.82),  # descriptor drop > .002
        ((0.501, 0.511, 0.520), 0.80, 0.82),  # mean gain < .001
    ],
)
def test_promotion_gate_fails_each_independent_requirement(
    candidate_scores: tuple[float, float, float],
    mean_descriptor: float,
    all_view: float,
) -> None:
    rows = {
        JOINT_CONTEXT_POOLING: _variant(
            (0.50, 0.51, 0.52),
            summary=0.95,
            mean_descriptor=0.80,
            all_view=0.82,
        ),
        SEPARATE_CONTEXT_POOLING: _variant(
            candidate_scores,
            summary=0.95,
            mean_descriptor=mean_descriptor,
            all_view=all_view,
        ),
    }
    assert authority.promotion_decision(rows, _gate())[
        "eligible_for_query_free_promotion"
    ] is False


def test_runner_shell_parses() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(
    not (LEGACY_ROOT / "caches/control_c256_geometric/train_shard1.pt").is_file(),
    reason="validated legacy Surface controls are not mounted",
)
def test_real_legacy_controls_remain_exactly_eligible() -> None:
    result = authority.validate_external_controls(
        external_root=LEGACY_ROOT,
        train_split=REPO_ROOT
        / "paper/artifacts/scannet_surface_region_query_free_train_scenes_20260731.txt",
        validation_split=REPO_ROOT
        / "paper/artifacts/scannet_surface_region_query_free_validation_scenes_20260731.txt",
        radio_checkpoint=Path(
            "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar"
        ),
        pfir_dev=REPO_ROOT
        / "radio_gs/benchmarks/scannet_pfir/split/scannet_pfir_small_v1_dev_candidates.txt",
        pfir_test=REPO_ROOT
        / "radio_gs/benchmarks/scannet_pfir/split/scannet_pfir_small_v1_test_candidates.txt",
    )
    assert result["status"] == "conditional_train01_prescreen_reuse_only"
    assert result["coverage"] == {
        "role": "train",
        "shards": [0, 1],
        "scenes": 16,
        "complete_four_shard_control": False,
        "claim": "validated_16_scene_prescreen_subset_only",
    }
    assert result["root_alias_resolved_once_then_descendants_nofollow"] is True
    assert [row["shard"] for row in result["controls"]] == [0, 1]
    assert [row["cache"]["sha256"] for row in result["controls"]] == [
        authority.LEGACY_CONTROL_SHA256[0],
        authority.LEGACY_CONTROL_SHA256[1],
    ]
