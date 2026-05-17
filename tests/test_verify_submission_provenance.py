from pathlib import Path

from radio_gs.scripts import verify_submission_provenance as verifier


def _complete_manifest() -> dict:
    return {
        "metadata": {"git_commit": "abc123", "git_dirty": True},
        "lerf": {
            "selector_policy": "fixed_threshold:0.60",
            "text_head": "SigLIP2",
            "teacher_model": "c-radio_v4-h",
            "evaluator": "eval_lerf_grounding",
            "evaluator_script": "radio_gs/scripts/eval_lerf_grounding.py",
            "evaluator_sha256": "abc",
            "rows": [
                {
                    "scene": "figurines",
                    "source": "output/reports/lerf_threshold_sweep.json",
                    "config": "radio_gs/configs/figurines.yaml",
                    "checkpoint": "output/figurines/checkpoints/latest.pth",
                    "feature_manifest": "output/radio_features_lerf/figurines",
                    "seed": 42,
                }
            ],
        },
        "scannet": {
            "selector_policy": "v67_teacherbalanced_gaussian_index_labelpoint",
            "text_head": "SigLIP2",
            "teacher_model": "c-radio_v4-h",
            "evaluator": "eval_scannet_pointcloud_radio_gs",
            "evaluator_script": "radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py",
            "evaluator_sha256": "def",
            "rows": [
                {
                    "scene": "scene0000_00",
                    "source": "output/scannet/result.json",
                    "config": "radio_gs/configs/scannet_scene0000_00.yaml",
                    "checkpoint": "output/scannet/checkpoints/best.pth",
                    "feature_manifest": "dataset/scannet_og/scene0000_00",
                    "seed": 42,
                }
            ],
        },
        "direct3d_readouts": [
            {
                "label": "VPR fixed threshold + RGB snap",
                "selector_policy": "fixed:thr0p25",
                "text_head": "SigLIP2",
                "evaluator": "eval_lerf_direct_3d_selection",
                "evaluator_script": "radio_gs/scripts/eval_lerf_direct_3d_selection.py",
                "evaluator_sha256": "ghi",
                "rows": [
                    {
                        "scene": "figurines",
                        "source": "output/direct3d/figurines/result.json",
                        "config": "radio_gs/configs/figurines.yaml",
                        "checkpoint": "output/figurines/checkpoints/best.pth",
                        "feature_manifest": "output/radio_features_lerf/figurines",
                        "teacher_model": "c-radio_v4-h",
                        "seed": 42,
                    }
                ],
            }
        ],
    }


def test_validate_manifest_accepts_complete_paper_rows() -> None:
    assert verifier.validate_manifest(_complete_manifest()) == []


def test_validate_manifest_reports_missing_row_fields() -> None:
    manifest = _complete_manifest()
    del manifest["lerf"]["rows"][0]["config"]
    del manifest["direct3d_readouts"][0]["rows"][0]["checkpoint"]

    issues = verifier.validate_manifest(manifest)

    assert "lerf.rows[0].config is missing" in issues
    assert "direct3d_readouts[0].rows[0].checkpoint is missing" in issues


def test_validate_manifest_reports_missing_global_git_commit() -> None:
    manifest = _complete_manifest()
    del manifest["metadata"]["git_commit"]

    issues = verifier.validate_manifest(manifest)

    assert "metadata.git_commit is missing" in issues


def test_validate_manifest_reports_missing_evaluator_hash() -> None:
    manifest = _complete_manifest()
    del manifest["scannet"]["evaluator_sha256"]

    issues = verifier.validate_manifest(manifest)

    assert "scannet.rows[0].evaluator_sha256 is missing" in issues


def _touch(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def test_validate_manifest_can_check_referenced_paths_exist(tmp_path: Path) -> None:
    manifest = _complete_manifest()
    for section in ("lerf", "scannet"):
        row = manifest[section]["rows"][0]
        _touch(tmp_path, row["source"])
        _touch(tmp_path, row["config"])
        _touch(tmp_path, row["checkpoint"])
        _touch(tmp_path, row["feature_manifest"])
        _touch(tmp_path, manifest[section]["evaluator_script"])
    direct = manifest["direct3d_readouts"][0]
    row = direct["rows"][0]
    _touch(tmp_path, row["source"])
    _touch(tmp_path, row["config"])
    _touch(tmp_path, row["checkpoint"])
    _touch(tmp_path, row["feature_manifest"])
    _touch(tmp_path, direct["evaluator_script"])

    assert verifier.validate_manifest(manifest, root=tmp_path, check_paths=True) == []


def test_validate_manifest_reports_missing_referenced_paths(tmp_path: Path) -> None:
    manifest = _complete_manifest()

    issues = verifier.validate_manifest(manifest, root=tmp_path, check_paths=True)

    assert "lerf.rows[0].source path does not exist: output/reports/lerf_threshold_sweep.json" in issues
    assert (
        "direct3d_readouts[0].rows[0].checkpoint path does not exist: "
        "output/figurines/checkpoints/best.pth"
    ) in issues
