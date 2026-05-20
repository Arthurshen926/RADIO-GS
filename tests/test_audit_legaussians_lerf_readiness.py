import json
from pathlib import Path

from radio_gs.scripts import audit_legaussians_lerf_readiness as audit


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _write_minimal_repo(root: Path, local_site: Path) -> None:
    for name in ("train.py", "render_mask.py", "eval.py"):
        _touch(root / name)
    _touch(local_site / "simple_knn" / "_C.fake.so")
    _touch(local_site / "diff_gaussian_rasterization" / "_C.fake.so")


def test_readiness_flags_missing_quantized_features(tmp_path):
    repo = tmp_path / "LEGaussians"
    local_site = tmp_path / "local_site"
    lerf_root = tmp_path / "lerf_ovs"
    _write_minimal_repo(repo, local_site)
    _touch(lerf_root / "ramen" / "images" / "frame_00001.jpg")
    _touch(lerf_root / "label" / "ramen" / "frame_00001.json")

    summary = audit.build_audit(
        repo=repo,
        local_site=local_site,
        lerf_root=lerf_root,
        scenes=["ramen"],
    )

    assert summary["repo_ready"] is True
    assert summary["all_scenes_ready"] is False
    assert summary["scenes"]["ramen"]["label_frame_count"] == 1
    assert summary["scenes"]["ramen"]["feature_preprocessing_complete"] is False
    assert "encoding indices" in summary["scenes"]["ramen"]["missing"][0]


def test_readiness_accepts_scene_with_codebook_and_indices(tmp_path):
    repo = tmp_path / "LEGaussians"
    local_site = tmp_path / "local_site"
    lerf_root = tmp_path / "lerf_ovs"
    _write_minimal_repo(repo, local_site)
    _touch(lerf_root / "figurines" / "images" / "frame_00041.jpg")
    _touch(lerf_root / "figurines" / "images" / "figurines_encoding_indices.pt")
    _touch(lerf_root / "figurines" / "images" / "figurines_codebook.pt")
    _touch(lerf_root / "label" / "figurines" / "frame_00041.json")

    summary = audit.build_audit(
        repo=repo,
        local_site=local_site,
        lerf_root=lerf_root,
        scenes=["figurines"],
    )

    scene = summary["scenes"]["figurines"]
    assert summary["all_scenes_ready"] is True
    assert scene["feature_preprocessing_complete"] is True
    assert scene["test_set"] == ["frame_00041"]
    assert scene["codebook_path"].endswith("figurines_codebook.pt")
    assert scene["encoding_indices_path"].endswith("figurines_encoding_indices.pt")


def test_cli_writes_json_and_markdown(tmp_path):
    repo = tmp_path / "LEGaussians"
    local_site = tmp_path / "local_site"
    lerf_root = tmp_path / "lerf_ovs"
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"
    _write_minimal_repo(repo, local_site)
    _touch(lerf_root / "ramen" / "images" / "frame_00001.jpg")
    _touch(lerf_root / "label" / "ramen" / "frame_00001.json")

    exit_code = audit.main(
        [
            "--repo",
            str(repo),
            "--local-site",
            str(local_site),
            "--lerf-root",
            str(lerf_root),
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
    assert payload["method"] == "LEGaussians"
    assert "| ramen | yes | 1 | no |" in markdown
