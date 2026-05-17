from __future__ import annotations

import json
from pathlib import Path

from radio_gs.scripts import build_train_feature_field_audit as audit


def _write_train_script(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_analyze_train_script_passes_core_audit_guards(tmp_path: Path) -> None:
    script = tmp_path / "train_feature_field.py"
    _write_train_script(
        script,
        "\n".join(
            [
                "from radio_gs.utils.checkpoint_io import load_trusted_checkpoint",
                "from radio_gs.data.benchmark_paths import resolve_split_feature_dir, resolve_split_frame_ids, resolve_split_pose_source",
                "def _get_git_metadata(): pass",
                "def _artifact_paths(): pass",
                "def _append_metrics_history(): pass",
                "def _write_run_manifest(): pass",
                "def _write_experiment_report(): pass",
                "def _acquire_training_lock():",
                "    lock_path = 'training.lock'",
                "def _release_training_lock(): pass",
                "def build_dataset(config):",
                "    train_feature_dir = resolve_split_feature_dir(config, 'train')",
                "    val_feature_dir = resolve_split_feature_dir(config, 'val')",
                "    train_pose = resolve_split_pose_source(config, 'train')",
                "    val_pose = resolve_split_pose_source(config, 'val')",
                "    train_frame_ids = resolve_split_frame_ids(config, 'train')",
                "    val_frame_ids = resolve_split_frame_ids(config, 'val')",
                "    return train_feature_dir, val_feature_dir, train_pose, val_pose, train_frame_ids, val_frame_ids",
                "def load_checkpoint(path):",
                "    return load_trusted_checkpoint(path)",
            ]
        ),
    )

    summary = audit.analyze_train_script(script, tests_root=tmp_path / "tests")
    checks = {row["id"]: row for row in summary["checks"]}

    assert checks["run_manifest"]["status"] == "pass"
    assert checks["split_resolution"]["status"] == "pass"
    assert checks["trusted_checkpoint_io"]["status"] == "pass"
    assert checks["training_lock"]["status"] == "pass"


def test_analyze_train_script_flags_missing_split_and_manifest_guards(tmp_path: Path) -> None:
    script = tmp_path / "train_feature_field.py"
    _write_train_script(
        script,
        "\n".join(
            [
                "def train():",
                "    pass",
                "def load_checkpoint(path):",
                "    return torch.load(path)",
            ]
        ),
    )

    summary = audit.analyze_train_script(script, tests_root=tmp_path / "tests")
    checks = {row["id"]: row for row in summary["checks"]}

    assert checks["run_manifest"]["status"] == "missing"
    assert checks["split_resolution"]["status"] == "missing"
    assert checks["trusted_checkpoint_io"]["status"] == "missing"
    assert summary["overall_status"] == "missing"


def test_analyze_train_script_accepts_wrapped_tensor_cache_loads(tmp_path: Path) -> None:
    script = tmp_path / "train_feature_field.py"
    _write_train_script(
        script,
        "\n".join(
            [
                "from radio_gs.utils.checkpoint_io import load_trusted_checkpoint",
                "from radio_gs.data.benchmark_paths import resolve_split_feature_dir, resolve_split_frame_ids, resolve_split_pose_source",
                "def load_training_tensor_cache(path, *, map_location='cpu', purpose='cache'):",
                "    return torch.load(path, map_location=map_location)",
                "def _get_git_metadata(): pass",
                "def _artifact_paths(): pass",
                "def _append_metrics_history(): pass",
                "def _write_run_manifest(): pass",
                "def _write_experiment_report(): pass",
                "def _acquire_training_lock():",
                "    lock_path = 'training.lock'",
                "def _release_training_lock(): pass",
                "def build_dataset(config):",
                "    train_feature_dir = resolve_split_feature_dir(config, 'train')",
                "    val_feature_dir = resolve_split_feature_dir(config, 'val')",
                "    train_pose = resolve_split_pose_source(config, 'train')",
                "    val_pose = resolve_split_pose_source(config, 'val')",
                "    train_frame_ids = resolve_split_frame_ids(config, 'train')",
                "    val_frame_ids = resolve_split_frame_ids(config, 'val')",
                "    return train_feature_dir, val_feature_dir, train_pose, val_pose, train_frame_ids, val_frame_ids",
                "def load_checkpoint(path):",
                "    return load_trusted_checkpoint(path)",
                "def load_feature(path):",
                "    return load_training_tensor_cache(path, purpose='radio_feature')",
            ]
        ),
    )

    summary = audit.analyze_train_script(script, tests_root=tmp_path / "tests")
    checks = {row["id"]: row for row in summary["checks"]}

    assert checks["raw_tensor_load_sites"]["status"] == "pass"
    assert "load_training_tensor_cache" in checks["raw_tensor_load_sites"]["evidence"]
    assert "Wrap or document raw torch.load feature/text/cache sites." not in summary["open_items"]


def test_repository_train_script_wraps_raw_tensor_cache_loads() -> None:
    summary = audit.analyze_train_script()
    checks = {row["id"]: row for row in summary["checks"]}

    assert checks["raw_tensor_load_sites"]["status"] == "pass"
    assert "load_training_tensor_cache" in checks["raw_tensor_load_sites"]["evidence"]


def test_repository_train_script_is_below_release_line_threshold() -> None:
    summary = audit.analyze_train_script()
    checks = {row["id"]: row for row in summary["checks"]}

    assert checks["script_size"]["status"] == "pass"
    assert summary["line_count"] <= 4000


def test_write_outputs_records_markdown_json_and_latex(tmp_path: Path) -> None:
    summary = {
        "script_path": "radio_gs/scripts/train_feature_field.py",
        "line_count": 6074,
        "overall_status": "risk",
        "checks": [
            {
                "id": "script_size",
                "status": "risk",
                "severity": "medium",
                "evidence": "6074 lines",
                "recommendation": "Split into train package modules.",
            }
        ],
        "test_coverage": [],
        "open_items": ["Split train_feature_field.py into modules."],
    }

    paths = audit.write_outputs(
        summary,
        tmp_path / "report.md",
        tmp_path / "report.json",
        tmp_path / "report.tex",
    )

    assert paths["markdown"].exists()
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["overall_status"] == "risk"
    assert "\\label{tab:train_feature_field_audit}" in paths["latex"].read_text(encoding="utf-8")
