import json
import subprocess

from radio_gs.scripts import audit_external_baselines as audit


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_inspect_lerf_assets_separates_direct_and_langsplat_partial(tmp_path):
    scene_root = tmp_path / "ramen"
    _touch(scene_root / "images" / "frame_00001.jpg")
    _touch(scene_root / "language_features" / "frame_00001_s.npy")
    _touch(scene_root / "language_features" / "frame_00001_f.npy")
    _touch(scene_root / "langsplat" / "language_features" / "frame_00001_s.npy")
    _touch(scene_root / "langsplat" / "language_features" / "frame_00001_f.npy")
    _touch(scene_root / "langsplat" / "language_features" / "frame_00002_s.npy")
    _touch(tmp_path / "label" / "ramen" / "gt" / "frame_00001" / "object.jpg")

    assets = audit.inspect_lerf_assets(tmp_path, scenes=("ramen",))

    assert assets["ramen"]["images"] == 1
    assert assets["ramen"]["direct_language_feature_masks"] == 1
    assert assets["ramen"]["direct_language_feature_vectors"] == 1
    assert assets["ramen"]["langsplat_language_feature_masks"] == 2
    assert assets["ramen"]["langsplat_language_feature_vectors"] == 1
    assert assets["ramen"]["direct_ready"] is True
    assert assets["ramen"]["langsplat_complete_pairs"] is False


def test_inspect_lerf_assets_requires_every_image_to_have_feature_pairs(tmp_path):
    scene_root = tmp_path / "ramen"
    _touch(scene_root / "images" / "frame_00001.jpg")
    _touch(scene_root / "images" / "frame_00002.jpg")
    _touch(scene_root / "language_features" / "frame_00001_s.npy")
    _touch(scene_root / "language_features" / "frame_00001_f.npy")
    _touch(scene_root / "langsplat" / "language_features" / "frame_00001_s.npy")
    _touch(scene_root / "langsplat" / "language_features" / "frame_00001_f.npy")

    assets = audit.inspect_lerf_assets(tmp_path, scenes=("ramen",))

    assert assets["ramen"]["direct_ready"] is False
    assert assets["ramen"]["langsplat_complete_pairs"] is False


def test_render_markdown_reports_commits_and_blockers():
    payload = {
        "baselines": [
            {
                "method": "OpenGaussian",
                "repo_path": "/tmp/OpenGaussian",
                "exists": True,
                "commit": "abc1234",
                "dirty": False,
                "submodule_missing": False,
                "blocker": "LERF language_features missing",
            }
        ],
        "lerf_assets": {
            "ramen": {
                "images": 131,
                "direct_language_feature_masks": 0,
                "direct_language_feature_vectors": 0,
                "langsplat_language_feature_masks": 18,
                "langsplat_language_feature_vectors": 18,
                "direct_ready": False,
                "langsplat_complete_pairs": True,
            }
        },
    }

    markdown = audit.render_markdown(payload)

    assert "OpenGaussian" in markdown
    assert "abc1234" in markdown
    assert "LERF language_features missing" in markdown
    assert "18/18" in markdown


def test_build_audit_includes_p1_reproduction_queue(tmp_path):
    payload = audit.build_audit(
        baselines_root=tmp_path / "baselines",
        lerf_root=tmp_path / "lerf",
        occam_output_root=tmp_path / "occam",
        artifact_root=tmp_path / "artifacts",
        semantic_metrics_path=tmp_path / "semantic_metrics.json",
    )

    methods = {row["method"] for row in payload["baselines"]}
    rows = {row["method"]: row for row in payload["baselines"]}

    assert {"LangSplat", "LEGaussians", "CAGS", "LaGa", "Semantic Gaussians"} <= methods
    assert "OpenGaFF" in methods
    assert rows["CAGS"]["url"] == "https://github.com/Wistzz/CAGS"
    assert "OpenGaussian-compatible LERF" in rows["CAGS"]["blocker"]
    assert "rasterizer ABI" in rows["CAGS"]["blocker"]
    assert "PyG source builds" in rows["CAGS"]["blocker"]
    assert "train.py/render_lerf_by_text.py reach CLI help" in rows["CAGS"]["blocker"]
    assert rows["LaGa"]["url"] == "https://github.com/SJTU-DeepVisionLab/LaGa"
    assert "view-dependent semantics" in rows["LaGa"]["blocker"]
    assert "kmeans_pytorch" in rows["LaGa"]["blocker"]
    assert rows["OpenGaFF"]["exists"] is False
    assert rows["OpenGaFF"]["url"] == "https://arxiv.org/abs/2605.06088"
    assert "code will be publicly released upon acceptance" in rows["OpenGaFF"]["blocker"]


def test_inspect_repo_updates_laga_blocker_when_kmeans_submodule_is_available(tmp_path):
    submodule_repo = tmp_path / "kmeans_pytorch"
    submodule_repo.mkdir()
    subprocess.run(["git", "init"], cwd=submodule_repo, check=True, stdout=subprocess.PIPE)
    (submodule_repo / "README.md").write_text("local kmeans fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=submodule_repo, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test User",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=submodule_repo,
        check=True,
        stdout=subprocess.PIPE,
    )

    repo = tmp_path / "LaGa"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(submodule_repo),
            "third_party/kmeans_pytorch",
        ],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    )

    row = audit.inspect_repo(
        repo,
        "LaGa",
        "https://github.com/SJTU-DeepVisionLab/LaGa",
        "Official implementation targets view-dependent semantics through object decomposition and descriptors; upstream gitlinks include a third_party/kmeans_pytorch path without a .gitmodules mapping on the current clone. Strict comparison needs affinity-feature training/inference notebook adaptation and same-evaluator exports.",
    )

    assert row["submodule_missing"] is False
    assert "kmeans_pytorch submodule is initialized" in row["blocker"]
    assert "train_scene.py/train_affinity_features.py reach CLI help" in row["blocker"]
    assert "laga_lerf_readiness_audit" in row["blocker"]
    assert "lack scene_point_cloud.ply, contrastive_feature_point_cloud.ply, and descriptor files" in row["blocker"]
    assert "robust GPU4/GPU5 follow-on chains are queued behind the Dr. Splat render/eval watchers" in row["blocker"]
    assert "render/eval completion marker" in row["blocker"]
    assert "12-field Occam RGB checkpoint restore compatibility patch" in row["blocker"]
    assert "missing .gitmodules mapping" not in row["blocker"]


def test_inspect_repo_updates_legaussians_blocker_when_preprocess_submodules_available(
    tmp_path,
):
    def create_source_repo(name):
        source_repo = tmp_path / name
        source_repo.mkdir()
        subprocess.run(["git", "init"], cwd=source_repo, check=True, stdout=subprocess.PIPE)
        (source_repo / "README.md").write_text(f"{name} fixture\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=source_repo, check=True, stdout=subprocess.PIPE)
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "user.name=Test User",
                "commit",
                "-m",
                "fixture",
            ],
            cwd=source_repo,
            check=True,
            stdout=subprocess.PIPE,
        )
        return source_repo

    sam_repo = create_source_repo("segment-anything")
    langsplat_sam_repo = create_source_repo("segment-anything-langsplat")
    repo = tmp_path / "LEGaussians"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
    for source_repo, path in (
        (sam_repo, "preprocess/segment-anything"),
        (langsplat_sam_repo, "preprocess/segment-anything-langsplat"),
    ):
        subprocess.run(
            [
                "git",
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                str(source_repo),
                path,
            ],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
        )

    row = audit.inspect_repo(
        repo,
        "LEGaussians",
        "https://github.com/buaavrcg/LEGaussians",
        "Official implementation requires dataset-specific preprocessing, local training/rendering, and evaluation before paper-table integration.",
    )

    assert row["submodule_missing"] is False
    assert "preprocess segment-anything gitlinks are initialized" in row["blocker"]
    assert "simple_knn/diff_gaussian_rasterization local site" in row["blocker"]
    assert "train.py reaches CLI help" in row["blocker"]
    assert "legaussians_lerf_readiness_audit" in row["blocker"]
    assert "four LERF scenes currently lack encoding indices/codebooks" in row["blocker"]
    assert "gpu_followon/lerf_compat_20260520" in row["blocker"]
    assert "after the LaGa phase in the robust follow-on chain" in row["blocker"]
    assert "LEGaussians render/eval watchers are queued" in row["blocker"]
    assert "official render_mask.py" in row["blocker"]
    assert "omits preprocess/segment-anything gitlinks" not in row["blocker"]


def test_inspect_repo_updates_gags_blocker_when_local_cli_smoke_is_available(tmp_path):
    repo = tmp_path / "GAGS"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)

    row = audit.inspect_repo(
        repo,
        "GAGS",
        "https://github.com/WHU-USI3DV/GAGS",
        "README releases code and labels but not pretrained/preprocessed models, so local feature extraction and training are required.",
    )

    assert row["submodule_missing"] is False
    assert "simple_knn/segment-anything local site" in row["blocker"]
    assert "train.py/render.py/evaluate_iou_loc.py reach CLI help" in row["blocker"]
    assert "ramen completed training/eval" in row["blocker"]
    assert "LocAcc 0.6479 / mIoU 0.4464" in row["blocker"]
    assert "figurines completed training/eval" in row["blocker"]
    assert "LocAcc 0.7321 / mIoU 0.4958" in row["blocker"]
    assert "teatime completed training and detached eval is running" in row["blocker"]
    assert "waldo_kitchen is now training" in row["blocker"]
    assert "All four scenes have detached eval watchers" in row["blocker"]
    assert "summarize_gags_lerf_baseline.py" in row["blocker"]
    assert "pretrained/preprocessed models are still unreleased" in row["blocker"]


def test_inspect_repo_updates_dr_splat_blocker_when_local_cli_smoke_is_available(
    tmp_path,
):
    repo = tmp_path / "Dr-Splat"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)

    row = audit.inspect_repo(
        repo,
        "Dr. Splat",
        "https://github.com/kaist-ami/Dr-Splat",
        "Official evaluation is marked TBA; fair comparison needs a local evaluator wrapper.",
    )

    assert row["submodule_missing"] is False
    assert "simple_knn/langsplat-rasterization/segment-anything local site" in row["blocker"]
    assert "train.py/render_activation.py reach CLI help" in row["blocker"]
    assert "ramen then teatime on GPU4" in row["blocker"]
    assert "figurines then waldo_kitchen on GPU5" in row["blocker"]
    assert "GPU4 retry chain is running after fixing the derived model_path directory creation bug" in row["blocker"]
    assert "chunked majority-voting accumulation" in row["blocker"]
    assert "eval_drsplat_lerf_masks.py" in row["blocker"]
    assert "--single_checkpoint" in row["blocker"]
    assert "render/eval watchers are queued" in row["blocker"]
    assert "drsplat_lerf_summary.{json,md}" in row["blocker"]
    assert "evaluation remains TBA upstream" in row["blocker"]


def test_inspect_repo_updates_langsplat_blocker_when_local_cli_smoke_is_available(
    tmp_path,
):
    repo = tmp_path / "LangSplat"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)

    row = audit.inspect_repo(
        repo,
        "LangSplat",
        "https://github.com/minghanqin/LangSplat",
        "Official implementation requires LERF/3D-OVS preprocessing or released checkpoints before a strict local macro can be reported.",
    )

    assert row["submodule_missing"] is False
    assert "simple_knn/langsplat-rasterization/segment-anything-langsplat local site" in row[
        "blocker"
    ]
    assert "train.py/render.py/evaluate_iou_loc.py reach CLI help" in row["blocker"]
    assert "All four scenes completed local compatibility training/render/eval" in row["blocker"]


def test_inspect_repo_updates_semantic_gaussians_blocker_when_partial_stack_is_available(
    tmp_path,
):
    repo = tmp_path / "semantic-gaussians"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)

    row = audit.inspect_repo(
        repo,
        "Semantic Gaussians",
        "https://github.com/sharinka0715/semantic-gaussians",
        "Official implementation targets ScanNet/MVImgNet-style semantic projection; needs a same-protocol local evaluator before comparison.",
    )

    assert row["submodule_missing"] is False
    assert "semantic_gaussians/local_site" in row["blocker"]
    assert "train.py/fusion.py/distill.py/eval_segmentation.py import successfully" in row["blocker"]
    assert "MinkowskiEngine 0.5.4 now imports successfully" in row["blocker"]
    assert "PyTorch-Encoding (`encoding`) now imports successfully" in row["blocker"]
    assert "semantic_gaussians_readiness_audit" in row["blocker"]
    assert "all four ScanNet scene zips are now extracted" in row["blocker"]
    assert "all four scenes have usable raw language features" in row["blocker"]
    assert "sega-py39-torch211-cu118" in row["blocker"]
    assert "requirements_after_minkowski_exit 0" in row["blocker"]
    assert "OpenSeg SavedModel weights were downloaded from the Stanford mirror" in row["blocker"]
    assert "full Semantic RGB 30k train chains are running" in row["blocker"]
    assert "label-PLY distill/eval watcher" in row["blocker"]
    assert "eval_label_ply.py" in row["blocker"]


def test_inspect_repo_flags_submodule_status_errors_as_missing(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        [
            "git",
            "update-index",
            "--add",
            "--cacheinfo",
            "160000",
            "6fdee8f2727f4506cfbbe553e23b895e27956588",
            "preprocess/segment-anything",
        ],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    )

    row = audit.inspect_repo(repo, "BrokenSubmoduleRepo", "https://example.invalid", "blocker")

    assert row["submodule_missing"] is True
    assert any("no submodule mapping" in line for line in row["submodule_status"])


def test_update_blockers_reflects_complete_lerf_assets():
    baselines = [
        {"method": "OpenGaussian", "blocker": "old"},
        {"method": "OccamLGS", "blocker": "old"},
        {"method": "GAGS", "blocker": "unchanged"},
    ]
    assets = {
        "ramen": {"direct_ready": True},
        "teatime": {"direct_ready": True},
    }

    updated = audit.update_blockers_for_lerf_assets(baselines, assets)

    assert "strict official-policy LERF" in updated[0]["blocker"]
    assert "build/train/eval" in updated[1]["blocker"]
    assert updated[2]["blocker"] == "unchanged"


def test_update_blockers_reflects_completed_occam_readout():
    baselines = [
        {"method": "OccamLGS", "blocker": "old"},
    ]
    assets = {
        "figurines": {"direct_ready": True},
        "ramen": {"direct_ready": True},
        "teatime": {"direct_ready": True},
        "waldo_kitchen": {"direct_ready": True},
    }
    occam_readout = {
        "complete": True,
        "objects": 208,
        "loc_acc": 0.8221153846153846,
        "miou": 0.45146960919709483,
    }

    updated = audit.update_blockers_for_lerf_assets(
        baselines,
        assets,
        occam_lerf_readout=occam_readout,
    )

    assert "all four LERF compatibility scenes completed" in updated[0]["blocker"]
    assert "LocAcc 0.8221 / mIoU 0.4515 over 208 objects" in updated[0]["blocker"]
    assert "pending" not in updated[0]["blocker"]


def test_update_blockers_reflects_completed_reproduction_summaries(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "gags_lerf_summary.json").write_text(
        json.dumps(
            {
                "scene_mean": {"locacc": 0.7, "miou": 0.4},
                "object_weighted": {"locacc": 0.8, "miou": 0.5, "query_count": 208},
            }
        ),
        encoding="utf-8",
    )
    (artifact_root / "drsplat_lerf_summary.json").write_text(
        json.dumps(
            {
                "macro": {
                    "miou": 0.17,
                    "acc025": 0.25,
                    "acc05": 0.11,
                    "count": 208,
                    "missing": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    (artifact_root / "legaussians_lerf_summary.json").write_text(
        json.dumps(
            {
                "scene_mean": {"miou": 0.26, "acc025": 0.39, "acc05": 0.23},
                "object_weighted": {"miou": 0.28, "count": 208, "missing": 0},
            }
        ),
        encoding="utf-8",
    )
    (artifact_root / "laga_lerf_summary.json").write_text(
        json.dumps(
            {
                "macro": {
                    "miou": 0.2,
                    "acc025": 0.3,
                    "acc05": 0.1,
                    "count": 208,
                    "missing": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    semantic_metrics = tmp_path / "semantic.json"
    semantic_metrics.write_text(
        json.dumps(
            {
                "metrics": {"mean_iou": 0.03},
                "scenes": {"scene0000_00": {"miou": 0.02}},
            }
        ),
        encoding="utf-8",
    )
    baselines = [
        {"method": "GAGS", "blocker": "old"},
        {"method": "Dr. Splat", "blocker": "old"},
        {"method": "LEGaussians", "blocker": "old"},
        {"method": "LaGa", "blocker": "old"},
        {"method": "Semantic Gaussians", "blocker": "old"},
    ]

    updated = audit.update_blockers_for_completed_reproductions(
        baselines,
        artifact_root=artifact_root,
        semantic_metrics_path=semantic_metrics,
    )
    rows = {row["method"]: row for row in updated}

    assert "scene-mean LocAcc 0.7000 / mIoU 0.4000" in rows["GAGS"]["blocker"]
    assert "mIoU 0.1700 / Acc@0.25 0.2500 / Acc@0.5 0.1100" in rows["Dr. Splat"]["blocker"]
    assert "render_mask.py compatibility pipeline completed" in rows["LEGaussians"]["blocker"]
    assert "descriptor building, mask export" in rows["LaGa"]["blocker"]
    assert "Mean IoU is 0.0300" in rows["Semantic Gaussians"]["blocker"]


def test_inspect_occam_lerf_readout_computes_object_weighted_macro(tmp_path):
    root = tmp_path / "occam"
    values = {
        "figurines": {"loc_acc": 0.75, "miou": 0.25, "objects": 2},
        "ramen": {"loc_acc": 1.0, "miou": 0.5, "objects": 6},
    }
    for scene, macro in values.items():
        path = root / f"occamlgs_{scene}_lerf_prerendered_eval_script.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"macro": macro}), encoding="utf-8")

    readout = audit.inspect_occam_lerf_readout(root, scenes=("figurines", "ramen"))

    assert readout["complete"] is True
    assert readout["objects"] == 8
    assert readout["loc_acc"] == 0.9375
    assert readout["miou"] == 0.4375


def test_write_audit_outputs_json_and_markdown(tmp_path):
    payload = {"baselines": [], "lerf_assets": {}}
    json_path = tmp_path / "audit.json"
    md_path = tmp_path / "audit.md"

    audit.write_audit(payload, json_path, md_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    assert "External Baseline Audit" in md_path.read_text(encoding="utf-8")


def test_link_complete_langsplat_features_only_when_all_images_have_pairs(tmp_path):
    scene_root = tmp_path / "ramen"
    for idx in range(1, 3):
        _touch(scene_root / "images" / f"frame_{idx:05d}.jpg")
        _touch(scene_root / "langsplat" / "language_features" / f"frame_{idx:05d}_s.npy")
        _touch(scene_root / "langsplat" / "language_features" / f"frame_{idx:05d}_f.npy")

    linked = audit.link_complete_langsplat_features(tmp_path, scenes=("ramen",))

    assert linked == {"ramen": True}
    assert (scene_root / "language_features").is_symlink()


def test_link_complete_langsplat_features_refuses_partial_scene(tmp_path):
    scene_root = tmp_path / "ramen"
    _touch(scene_root / "images" / "frame_00001.jpg")
    _touch(scene_root / "images" / "frame_00002.jpg")
    _touch(scene_root / "langsplat" / "language_features" / "frame_00001_s.npy")
    _touch(scene_root / "langsplat" / "language_features" / "frame_00001_f.npy")

    linked = audit.link_complete_langsplat_features(tmp_path, scenes=("ramen",))

    assert linked == {"ramen": False}
    assert not (scene_root / "language_features").exists()


def test_parser_exposes_link_complete_langsplat_flag():
    parser = audit.build_arg_parser()
    args = parser.parse_args(["--link-complete-langsplat"])

    assert args.link_complete_langsplat is True
