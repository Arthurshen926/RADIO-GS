import json
import zipfile
from pathlib import Path

from radio_gs.scripts import audit_semantic_gaussians_readiness as audit


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _write_minimal_repo(root: Path, local_site: Path) -> None:
    for name in ("train.py", "fusion.py", "distill.py", "eval_segmentation.py", "view_viser.py"):
        _touch(root / name)
    for name in ("official_train.yaml", "fusion_scannet.yaml", "distill_scannet.yaml", "eval.yaml"):
        _touch(root / "config" / name)
    for module in ("simple_knn", "rgbd_rasterization", "channel_rasterization"):
        _touch(local_site / module / "_C.fake.so")
    _touch(local_site / "segment_anything" / "__init__.py")
    for module in ("tensorflow", "viser"):
        _touch(local_site / module / "__init__.py")


def _write_zip(path: Path, name: str = "features.pt") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, b"x")


def test_readiness_flags_missing_compiled_dependencies_and_extracted_scannet(tmp_path):
    repo = tmp_path / "semantic-gaussians"
    local_site = tmp_path / "local_site"
    scannet_root = tmp_path / "scannet"
    output_root = tmp_path / "outputs"
    _write_minimal_repo(repo, local_site)
    _touch(scannet_root / "scene0000_00.zip")

    summary = audit.build_audit(
        repo=repo,
        local_site=local_site,
        scannet_root=scannet_root,
        output_root=output_root,
        scenes=["scene0000_00"],
    )

    assert summary["repo_ready"] is True
    assert summary["native_ready"] is True
    assert summary["dependency_ready"] is False
    assert summary["environment_ready"] is False
    assert summary["strict_ready"] is False
    assert summary["dependencies"]["MinkowskiEngine"] is False
    assert summary["dependencies"]["encoding"] is False
    scene = summary["scenes"]["scene0000_00"]
    assert scene["zip_present"] is True
    assert scene["extracted_ready"] is False
    assert scene["ready_for_eval"] is False
    assert "missing extracted ScanNet scene directory" in scene["missing"]


def test_readiness_accepts_complete_semantic_gaussians_scene(tmp_path):
    repo = tmp_path / "semantic-gaussians"
    local_site = tmp_path / "local_site"
    scannet_root = tmp_path / "scannet"
    output_root = tmp_path / "outputs"
    _write_minimal_repo(repo, local_site)
    for module in ("MinkowskiEngine", "encoding"):
        _touch(local_site / module / "__init__.py")
    for relative in ("color/frame.jpg", "pose/frame.txt", "intrinsic/intrinsic_color.txt", "points3d.ply"):
        _touch(scannet_root / "scene0000_00" / relative)
    _touch(scannet_root / "scene0000_00" / "language_features" / "features.pt")
    _touch(output_root / "scene0000_00" / "gaussians" / "point_cloud" / "iteration_30000" / "point_cloud.ply")
    _touch(output_root / "scene0000_00" / "fusion" / "features.pt")
    _touch(output_root / "scene0000_00" / "distill" / "checkpoint.pth")
    _touch(output_root / "scene0000_00" / "eval" / "metrics.json")

    summary = audit.build_audit(
        repo=repo,
        local_site=local_site,
        scannet_root=scannet_root,
        output_root=output_root,
        scenes=["scene0000_00"],
    )

    assert summary["all_scenes_ready"] is True
    assert summary["strict_ready"] is True
    scene = summary["scenes"]["scene0000_00"]
    assert scene["ready_for_eval"] is True
    assert scene["extracted_ready"] is True
    assert scene["fusion_ready"] is True
    assert scene["distill_ready"] is True


def test_readiness_accepts_dependencies_from_separate_site(tmp_path):
    repo = tmp_path / "semantic-gaussians"
    local_site = tmp_path / "local_site"
    dependency_site = tmp_path / "dependency_site"
    scannet_root = tmp_path / "scannet"
    output_root = tmp_path / "outputs"
    _write_minimal_repo(repo, local_site)
    for module in ("MinkowskiEngine", "encoding"):
        _touch(dependency_site / module / "__init__.py")

    summary = audit.build_audit(
        repo=repo,
        local_site=local_site,
        dependency_site=dependency_site,
        scannet_root=scannet_root,
        output_root=output_root,
        scenes=["scene0000_00"],
    )

    assert summary["native_ready"] is True
    assert summary["dependencies"]["MinkowskiEngine"] is True
    assert summary["dependencies"]["encoding"] is True
    assert summary["dependency_ready"] is True
    assert summary["environment_ready"] is True


def test_readiness_promotes_completed_scene_outputs_even_if_current_import_stack_is_missing(tmp_path):
    repo = tmp_path / "semantic-gaussians"
    local_site = tmp_path / "local_site"
    scannet_root = tmp_path / "scannet"
    output_root = tmp_path / "outputs"
    _write_minimal_repo(repo, local_site)
    for relative in ("color/frame.jpg", "pose/frame.txt", "intrinsic/intrinsic_color.txt", "points3d.ply"):
        _touch(scannet_root / "scene0000_00" / relative)
    _touch(scannet_root / "scene0000_00" / "language_features" / "features.pt")
    _touch(output_root / "scene0000_00" / "gaussians" / "point_cloud" / "iteration_30000" / "point_cloud.ply")
    _touch(output_root / "scene0000_00" / "fusion" / "features.pt")
    _touch(output_root / "scene0000_00" / "distill" / "checkpoint.pth")
    _touch(output_root / "scene0000_00" / "eval" / "metrics.json")

    summary = audit.build_audit(
        repo=repo,
        local_site=local_site,
        scannet_root=scannet_root,
        output_root=output_root,
        scenes=["scene0000_00"],
    )

    assert summary["dependency_ready"] is False
    assert summary["all_scenes_ready"] is True
    assert summary["strict_ready"] is False
    assert summary["next_action"] == "Promote completed ScanNet metrics into the external baseline registry."


def test_readiness_rejects_dependency_package_that_fails_import(tmp_path):
    repo = tmp_path / "semantic-gaussians"
    local_site = tmp_path / "local_site"
    scannet_root = tmp_path / "scannet"
    output_root = tmp_path / "outputs"
    _write_minimal_repo(repo, local_site)
    broken_init = local_site / "tensorflow" / "__init__.py"
    broken_init.write_text("raise ImportError('protobuf mismatch')\n", encoding="utf-8")

    summary = audit.build_audit(
        repo=repo,
        local_site=local_site,
        scannet_root=scannet_root,
        output_root=output_root,
        scenes=["scene0000_00"],
    )

    assert summary["dependencies"]["tensorflow"] is False


def test_readiness_accepts_blender_style_extracted_scene_data(tmp_path):
    repo = tmp_path / "semantic-gaussians"
    local_site = tmp_path / "local_site"
    scannet_root = tmp_path / "scannet"
    output_root = tmp_path / "outputs"
    _write_minimal_repo(repo, local_site)
    _touch(scannet_root / "scene0062_00" / "color" / "000000.jpg")
    _touch(scannet_root / "scene0062_00" / "transforms_train.json")
    _touch(scannet_root / "scene0062_00" / "transforms_test.json")
    _touch(scannet_root / "scene0062_00" / "points3d.ply")

    summary = audit.build_audit(
        repo=repo,
        local_site=local_site,
        scannet_root=scannet_root,
        output_root=output_root,
        scenes=["scene0062_00"],
    )

    assert summary["scenes"]["scene0062_00"]["extracted_ready"] is True


def test_readiness_flags_missing_or_invalid_raw_language_features(tmp_path):
    repo = tmp_path / "semantic-gaussians"
    local_site = tmp_path / "local_site"
    scannet_root = tmp_path / "scannet"
    output_root = tmp_path / "outputs"
    _write_minimal_repo(repo, local_site)
    _touch(scannet_root / "scene0000_00" / "color" / "000000.jpg")
    _touch(scannet_root / "scene0000_00" / "transforms_train.json")
    _touch(scannet_root / "scene0000_00" / "transforms_test.json")
    _touch(scannet_root / "scene0000_00" / "points3d.ply")
    _touch(scannet_root / "scene0000_00" / "language_features.zip")
    _write_zip(scannet_root / "scene0062_00" / "language_features.zip")

    summary = audit.build_audit(
        repo=repo,
        local_site=local_site,
        scannet_root=scannet_root,
        output_root=output_root,
        scenes=["scene0000_00", "scene0062_00"],
    )

    scene0000 = summary["scenes"]["scene0000_00"]
    scene0062 = summary["scenes"]["scene0062_00"]
    assert scene0000["language_features_ready"] is False
    assert "missing extracted or valid raw language features" in scene0000["missing"]
    assert scene0062["language_features_ready"] is True


def test_cli_writes_json_and_markdown(tmp_path):
    repo = tmp_path / "semantic-gaussians"
    local_site = tmp_path / "local_site"
    scannet_root = tmp_path / "scannet"
    output_root = tmp_path / "outputs"
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"
    _write_minimal_repo(repo, local_site)
    _touch(scannet_root / "scene0000_00.zip")

    exit_code = audit.main(
        [
            "--repo",
            str(repo),
            "--local-site",
            str(local_site),
            "--scannet-root",
            str(scannet_root),
            "--output-root",
            str(output_root),
            "--scenes",
            "scene0000_00",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    markdown = out_md.read_text(encoding="utf-8")
    assert payload["method"] == "Semantic Gaussians"
    assert "MinkowskiEngine" in markdown
    assert "| scene0000_00 | yes | no | no | no | no | no |" in markdown


def test_cli_accepts_legacy_out_json_and_out_md_aliases(tmp_path):
    repo = tmp_path / "semantic-gaussians"
    local_site = tmp_path / "local_site"
    scannet_root = tmp_path / "scannet"
    output_root = tmp_path / "outputs"
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"
    _write_minimal_repo(repo, local_site)
    _touch(scannet_root / "scene0000_00.zip")

    exit_code = audit.main(
        [
            "--repo",
            str(repo),
            "--local-site",
            str(local_site),
            "--scannet-root",
            str(scannet_root),
            "--output-root",
            str(output_root),
            "--scenes",
            "scene0000_00",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert exit_code == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["method"] == "Semantic Gaussians"
    assert out_md.read_text(encoding="utf-8").startswith("# Semantic Gaussians Readiness Audit")
