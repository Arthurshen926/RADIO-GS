from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE = (
    REPO_ROOT
    / "radio_gs/scripts/run_scannet_canonical_mpr_v3_paper8_reconstruction.sh"
)
VALIDATOR = (
    REPO_ROOT
    / "radio_gs/scripts/validate_scannet_canonical_mpr_v3_paper8_stage.py"
)


def _source() -> str:
    return QUEUE.read_text(encoding="utf-8")


def _validator_module():
    spec = importlib.util.spec_from_file_location("paper8_stage_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_queue_freezes_exact_missing_scene_geometry_carriers() -> None:
    source = _source()
    expected = {
        "0000": (
            "scannet_og_scene0000_00_v14/checkpoints/best.pth",
            "d5ce0a13264ee2bb5a638a2eab6c51cd8a81e87ec1b03d8a70b956c1d08c40fa",
            "1b56ee37ad05045c6b97bc24ee34dce59a06b84486abb0735a87f887af843fbc",
        ),
        "0070": (
            "scannet_og_scene0070_00_v14_b1/checkpoints/best.pth",
            "2e083e662b8be14dcb5c02c58e3b34ea893294f82c9f583e1f1dad571e7437d1",
            "0b13a36666b63d4d47abc2d841065c98ed4d3229eb6de6833b69aa8ea7f83068",
        ),
        "0097": (
            "scannet_og_scene0097_00_v14/checkpoints/best.pth",
            "28e00ac88cf6cb1a8687d0ac44e9b3cc025910c24a7060a6850b6f62e669079c",
            "cb8d41739f663c807c81ee99e6e2f78411a3a09e3f79b3ac130514b6b6f76d96",
        ),
        "0347": (
            "scannet_og_scene0347_00_v14/checkpoints/best.pth",
            "d2b33c0ebabeba9e245c6fae358762a5e9c9172e964f8331890b2cc098564103",
            "5406f3c9e8b74966d17a54d581aeec6ecf03a08e640e40c4b12400fe1f23508f",
        ),
        "0400": (
            "scannet_og_scene0400_00_v67_dino_cv001_b2_s32768_ft20/checkpoints/best.pth",
            "e3bb13d1ea1e7baade004873e0db2f261b7a4c7eb2e56c6636e1e3ca11113db4",
            "dad197a9fb6ad1c104e16b8cd0a2879f337d2029f1c0167cf813f902a9084d7c",
        ),
        "0590": (
            "scannet_og_scene0590_00_v14/checkpoints/best.pth",
            "f6aef423b88a5a13c9e58687448616a2bbbca03ad080c27ba2e212b78ead587b",
            "c514cbdc09657a6ba865d45c1aca8d2f7b0b70994ea644359b5cc2e62749831c",
        ),
    }
    for scene, (checkpoint, digest, feature_digest) in expected.items():
        assert f"  {scene})" in source
        assert checkpoint in source
        assert digest in source
        assert feature_digest in source
    assert "EXPECTED_EXCLUDED_STEMS=\"60,80,1260\"" in source
    assert "scannet_og_scene0400_00_v14/checkpoints/best.pth" not in source


def test_queue_uses_frozen_current_field_ladder_and_no_legacy_semantic_stage() -> None:
    source = _source()
    required = [
        "select_fidelity_validation_frames.py",
        "build_gaussian_multiview_teacher_cache.py",
        "--feature-space radio",
        "--feature-space dino_v3",
        "--feature-space sam3",
        "--capability-map-source project_raw",
        "train_canonical_radio_field.py",
        "--coefficient-dim 256",
        "--local-dim 128",
        "--primitive-fusion",
        "--official-capability-loss",
        "--epochs 20",
        "finetune_canonical_radio_rendering.py",
        "--steps 256",
        "--mpr-weight 0.10",
        "--max-mpr-drop 0.005",
        "--dino-render-weight 0.20",
        "--sam3-render-weight 0.20",
        "--capability-local-affinity-weight 0.25",
        "--selection-policy capability_pareto",
        "--max-capability-drop 0.002",
        "build_canonical_capability_views.py",
        "build_canonical_support_graph.py",
        "--neighbors 16",
        "--capability-affinity-mode signed_hash",
        "--affinity-dim 256",
        "--topology-mode symmetric_union",
    ]
    for token in required:
        assert token in source
    assert "--train-fusion" not in source
    assert "--train-basis" not in source
    assert "build_surface_region_semantic_cache.py" not in source
    assert "semantic-teacher-root" not in source


def test_queue_binds_every_formal_source_and_uses_moderate_300w_guard() -> None:
    source = _source()
    for token in (
        "--expected-feature-output-bundle-sha256",
        "--expected-geometry-checkpoint-sha256",
        "--expected-responsibility-cache-sha256",
        "--expected-mpr-cache-sha256",
        "--expected-dino-v3-mpr-cache-sha256",
        "--expected-sam3-mpr-cache-sha256",
        "--expected-radio-checkpoint-sha256",
        "run_with_gpu_thermal_guard.sh",
        "GPU_MAX_POWER_LIMIT_W=\"${GPU_MAX_POWER_LIMIT_W:-300.5}\"",
        "GPU_POLL_SECONDS=\"${GPU_POLL_SECONDS:-20}\"",
        "GPU_SOFT_PAUSE_TEMP_C=\"${GPU_SOFT_PAUSE_TEMP_C:-81}\"",
        "GPU_SOFT_RESUME_TEMP_C=\"${GPU_SOFT_RESUME_TEMP_C:-76}\"",
        "GPU_MAX_TEMP_C=\"${GPU_MAX_TEMP_C:-84}\"",
        "GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES=\"${GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES:-3}\"",
        "exclusive-singleton-after-clear-v1",
    ):
        assert token in source


def test_plan_validator_rejects_query_or_mask_selected_plan(tmp_path: Path) -> None:
    module = _validator_module()
    manifest = tmp_path / "frame_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "policy": module.PLAN_POLICY,
        "feature_dir": str(tmp_path),
        "frame_manifest": str(manifest),
        "frame_manifest_sha256": module._sha256(manifest),
        "validation_frame_ids": [20, 40, 60, 80],
        "requested_validation_views": 4,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "query_opened": True,
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(payload), encoding="utf-8")
    try:
        module.validate_plan(plan)
    except ValueError as exc:
        assert "query_opened" in str(exc)
    else:
        raise AssertionError("query-opened validation plan was accepted")
