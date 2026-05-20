import json
from pathlib import Path

from radio_gs.scripts import audit_laga_lerf_readiness as audit


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _write_minimal_repo(root: Path, local_site: Path) -> None:
    for name in ("train_scene.py", "train_affinity_features.py", "render.py", "inference.ipynb"):
        _touch(root / name)
    _touch(local_site / "simple_knn" / "_C.fake.so")
    _touch(local_site / "diff_gaussian_rasterization" / "_C.fake.so")
    _touch(local_site / "diff_gaussian_rasterization_contrastive_f" / "_C.fake.so")


def test_readiness_flags_missing_laga_stage_outputs(tmp_path):
    repo = tmp_path / "LaGa"
    local_site = tmp_path / "local_site"
    lerf_root = tmp_path / "lerf_ovs"
    model_root = tmp_path / "models"
    _write_minimal_repo(repo, local_site)
    _touch(lerf_root / "ramen" / "images" / "frame_00001.jpg")
    _touch(lerf_root / "ramen" / "sparse" / "0" / "cameras.bin")
    _touch(lerf_root / "label" / "ramen" / "frame_00001.json")

    summary = audit.build_audit(
        repo=repo,
        local_site=local_site,
        lerf_root=lerf_root,
        model_root=model_root,
        scenes=["ramen"],
    )

    assert summary["repo_ready"] is True
    assert summary["all_scenes_ready"] is False
    scene = summary["scenes"]["ramen"]
    assert scene["data_ready"] is True
    assert scene["scene_checkpoint_ready"] is False
    assert scene["affinity_checkpoint_ready"] is False
    assert scene["descriptor_ready"] is False
    assert "scene_point_cloud.ply" in scene["missing"][0]


def test_readiness_accepts_complete_laga_scene_outputs(tmp_path):
    repo = tmp_path / "LaGa"
    local_site = tmp_path / "local_site"
    lerf_root = tmp_path / "lerf_ovs"
    model_root = tmp_path / "models"
    iter_dir = model_root / "figurines" / "point_cloud" / "iteration_30000"
    _write_minimal_repo(repo, local_site)
    _touch(lerf_root / "figurines" / "images" / "frame_00041.jpg")
    _touch(lerf_root / "figurines" / "sparse" / "0" / "cameras.bin")
    _touch(lerf_root / "label" / "figurines" / "frame_00041.json")
    _touch(iter_dir / "scene_point_cloud.ply")
    _touch(iter_dir / "contrastive_feature_point_cloud.ply")
    _touch(iter_dir / "multi_lvl_cluster_features.pth")
    _touch(iter_dir / "multi_lvl_cluster_feature_weights.pth")
    _touch(iter_dir / "multi_lvl_seg_scores.pth")

    summary = audit.build_audit(
        repo=repo,
        local_site=local_site,
        lerf_root=lerf_root,
        model_root=model_root,
        scenes=["figurines"],
    )

    scene = summary["scenes"]["figurines"]
    assert summary["all_scenes_ready"] is True
    assert scene["ready_for_same_protocol_export"] is True
    assert scene["test_set"] == ["frame_00041"]


def test_readiness_accepts_split_scene_and_affinity_iterations(tmp_path):
    repo = tmp_path / "LaGa"
    local_site = tmp_path / "local_site"
    lerf_root = tmp_path / "lerf_ovs"
    model_root = tmp_path / "models"
    scene_iter_dir = model_root / "ramen" / "point_cloud" / "iteration_30001"
    affinity_iter_dir = model_root / "ramen" / "point_cloud" / "iteration_30000"
    _write_minimal_repo(repo, local_site)
    _touch(lerf_root / "ramen" / "images" / "frame_00001.jpg")
    _touch(lerf_root / "ramen" / "sparse" / "0" / "cameras.bin")
    _touch(lerf_root / "label" / "ramen" / "frame_00001.json")
    _touch(scene_iter_dir / "scene_point_cloud.ply")
    _touch(affinity_iter_dir / "contrastive_feature_point_cloud.ply")
    _touch(affinity_iter_dir / "multi_lvl_cluster_features.pth")
    _touch(affinity_iter_dir / "multi_lvl_cluster_feature_weights.pth")
    _touch(affinity_iter_dir / "multi_lvl_seg_scores.pth")

    summary = audit.build_audit(
        repo=repo,
        local_site=local_site,
        lerf_root=lerf_root,
        model_root=model_root,
        scenes=["ramen"],
        scene_iteration=30001,
        affinity_iteration=30000,
    )

    scene = summary["scenes"]["ramen"]
    assert summary["all_scenes_ready"] is True
    assert summary["scene_iteration"] == 30001
    assert summary["affinity_iteration"] == 30000
    assert scene["scene_checkpoint_ready"] is True
    assert scene["affinity_checkpoint_ready"] is True
    assert scene["descriptor_ready"] is True


def test_cli_writes_json_and_markdown(tmp_path):
    repo = tmp_path / "LaGa"
    local_site = tmp_path / "local_site"
    lerf_root = tmp_path / "lerf_ovs"
    model_root = tmp_path / "models"
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"
    _write_minimal_repo(repo, local_site)
    _touch(lerf_root / "ramen" / "images" / "frame_00001.jpg")
    _touch(lerf_root / "ramen" / "sparse" / "0" / "cameras.bin")
    _touch(lerf_root / "label" / "ramen" / "frame_00001.json")

    exit_code = audit.main(
        [
            "--repo",
            str(repo),
            "--local-site",
            str(local_site),
            "--lerf-root",
            str(lerf_root),
            "--model-root",
            str(model_root),
            "--scenes",
            "ramen",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    markdown = out_md.read_text(encoding="utf-8")
    assert payload["method"] == "LaGa"
    assert "| ramen | yes | 1 | no | no | no |" in markdown
