#!/usr/bin/env python3
"""Audit external open-vocabulary 3DGS baselines and local LERF assets."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Iterable


LERF_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")

BASELINES = (
    {
        "method": "OpenGaussian",
        "repo_dir": "OpenGaussian",
        "url": "https://github.com/yanmin-wu/OpenGaussian",
        "blocker": "ScanNet is reproduced locally; four-scene LERF language_features are available, but a strict official-policy LERF training/evaluation rerun is pending.",
    },
    {
        "method": "LangSplatV2",
        "repo_dir": "LangSplatV2",
        "url": "https://github.com/ZhaoYujie2002/LangSplatV2",
        "blocker": "Local LERF compatibility reruns completed all four scenes through all three feature levels plus eval_lerf.py --quick_render; current summary is LocAcc 0.6176 / mIoU 0.4601 scene-mean and LocAcc 0.6010 / mIoU 0.4487 object-weighted over 208 queries. This remains a compatibility rerun, not a strict released-checkpoint macro.",
    },
    {
        "method": "OccamLGS",
        "repo_dir": "OccamLGS",
        "url": "https://github.com/insait-institute/OccamLGS",
        "blocker": "all four LERF compatibility scenes completed RGB training, language-feature extraction, test feature-map rendering, and tracked normalized pre-rendered readout: LocAcc 0.8221 / mIoU 0.4515 over 208 objects. This remains a compatibility readout, not a strict released-checkpoint macro.",
    },
    {
        "method": "GAGS",
        "repo_dir": "GAGS",
        "url": "https://github.com/WHU-USI3DV/GAGS",
        "blocker": "README releases code and labels but not pretrained/preprocessed models, so local feature extraction and training are required.",
    },
    {
        "method": "Dr. Splat",
        "repo_dir": "Dr-Splat",
        "url": "https://github.com/kaist-ami/Dr-Splat",
        "blocker": "Official evaluation is marked TBA; fair comparison needs a local evaluator wrapper.",
    },
    {
        "method": "LangSplat",
        "repo_dir": "LangSplat",
        "url": "https://github.com/minghanqin/LangSplat",
        "blocker": "Official implementation requires LERF/3D-OVS preprocessing or released checkpoints before a strict local macro can be reported.",
    },
    {
        "method": "LEGaussians",
        "repo_dir": "LEGaussians",
        "url": "https://github.com/buaavrcg/LEGaussians",
        "blocker": "Official implementation requires dataset-specific preprocessing, local training/rendering, and evaluation before paper-table integration.",
    },
    {
        "method": "CAGS",
        "repo_dir": "CAGS",
        "url": "https://github.com/Wistzz/CAGS",
        "blocker": "OpenGaussian-compatible LERF training/render/eval scripts are available; the rasterizer ABI blocker and PyG source builds are cleared locally via output/baselines/cags/local_site, and train.py/render_lerf_by_text.py reach CLI help. Strict local reproduction still needs data-path setup and same-evaluator metric export. ScanNet evaluation is marked TODO upstream.",
    },
    {
        "method": "Semantic Gaussians",
        "repo_dir": "semantic-gaussians",
        "url": "https://github.com/sharinka0715/semantic-gaussians",
        "blocker": "Official implementation targets ScanNet/MVImgNet-style semantic projection; needs a same-protocol local evaluator before comparison.",
    },
    {
        "method": "LaGa",
        "repo_dir": "LaGa",
        "url": "https://github.com/SJTU-DeepVisionLab/LaGa",
        "blocker": "Official implementation targets view-dependent semantics through object decomposition and descriptors; upstream gitlinks include a third_party/kmeans_pytorch path without a .gitmodules mapping on the current clone. Strict comparison needs affinity-feature training/inference notebook adaptation and same-evaluator exports.",
    },
    {
        "method": "OpenGaFF",
        "repo_dir": "OpenGaFF",
        "url": "https://arxiv.org/abs/2605.06088",
        "blocker": "The arXiv paper reports state-of-the-art LERF-OVS and ScanNet context numbers, but its source states that code will be publicly released upon acceptance; no public implementation was found in the arXiv metadata/source or web search on 2026-05-18. Keep as published context, not a reproducible local baseline.",
    },
)


def _run_git_result(repo: Path, args: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.returncode, result.stdout.strip()


def _run_git(repo: Path, args: list[str]) -> str:
    return _run_git_result(repo, args)[1]


def _blocker_for_repo_state(method: str, blocker: str, submodule_missing: bool | None) -> str:
    if method == "GAGS" and submodule_missing is False:
        return (
            "README releases code and labels, and the local simple_knn/segment-anything "
            "local site is built under output/baselines/gags/local_site; "
            "train.py/render.py/evaluate_iou_loc.py reach CLI help. The integer "
            "SAM-seg-map resize fix, optional mediapy image-writer fallback, and "
            "full-camera-list label-eval guard are in place. Full LERF compatibility "
            "training/eval is in flight on GPUs 4/5: ramen completed training/eval "
            "with LocAcc 0.6479 / mIoU 0.4464 at mask_thresh 0.4, "
            "figurines completed training/eval with LocAcc 0.7321 / mIoU 0.4958 "
            "at mask_thresh 0.4, teatime completed training and detached eval is "
            "running on GPU4, and waldo_kitchen is now training from an Occam 30k "
            "RGB start on GPU5. All four "
            "scenes have detached eval watchers plus a final "
            "summarize_gags_lerf_baseline.py watcher for gags_lerf_summary.{json,md}. "
            "Strict metric rows remain pending until those jobs finish because "
            "pretrained/preprocessed models are still unreleased."
        )
    if method == "Dr. Splat" and submodule_missing is False:
        return (
            "The local simple_knn/langsplat-rasterization/segment-anything local site "
            "is built under output/baselines/dr_splat/local_site; "
            "train.py/render_activation.py reach CLI help. Four Occam-start, PQ-index "
            "majority-voting LERF compatibility jobs are queued behind the GAGS GPU4/5 "
            "chains: ramen then teatime on GPU4, and figurines then waldo_kitchen on "
            "GPU5. The GPU4 retry chain is running after fixing the derived "
            "model_path directory creation bug in the official train.py path; "
            "chunked majority-voting accumulation is now used to avoid the "
            "teatime dense mask-by-Gaussian CPU allocation. "
            "A local nested-mask same-protocol evaluator is available at "
            "radio_gs/scripts/eval_drsplat_lerf_masks.py, and the VALA Dr. Splat "
            "render entry now reaches CLI help with --single_checkpoint for chkpnt0.pth "
            "mask rendering. Detached render/eval watchers are queued after the Dr. "
            "Splat GPU4/5 chains, followed by a final drsplat_lerf_summary.{json,md} "
            "watcher. Fair table integration still needs those local outputs to finish "
            "because evaluation remains TBA upstream."
        )
    if method == "LangSplat" and submodule_missing is False:
        return (
            "The local simple_knn/langsplat-rasterization/segment-anything-langsplat "
            "local site is built with NumPy pinned to 1.26.4; "
            "train.py/render.py/evaluate_iou_loc.py reach CLI help. All four scenes "
            "completed local compatibility training/render/eval after fp32 dim-3 "
            "feature conversion, chunked decoder eval, and split-aware train/test "
            "feature path fixes. Current summary: scene-mean LocAcc 0.7335 / "
            "mIoU 0.4433 and object-weighted LocAcc 0.7356 / mIoU 0.4613 over "
            "208 queries. This remains a compatibility rerun, not a strict "
            "released-checkpoint macro."
        )
    if method == "LEGaussians" and submodule_missing is False:
        return (
            "Official implementation requires local feature preprocessing, training, "
            "rendering, and evaluation before paper-table integration. The local "
            "preprocess segment-anything gitlinks are initialized through repaired "
            ".gitmodules entries, so recursive submodule status is clean; the "
            "simple_knn/diff_gaussian_rasterization local site is built under "
            "output/baselines/legaussians/local_site and train.py reaches CLI help; "
            "paper/artifacts/legaussians_lerf_readiness_audit.{json,md} now records "
            "the exact LERF readiness state. All four LERF scenes currently lack "
            "encoding indices/codebooks from LEGaussians preprocessing, so strict "
            "comparison still requires dense feature quantization before training, "
            "rendering, and same-evaluator metric export. Robust GPU4/GPU5 follow-on "
            "chains are queued under output/baselines/gpu_followon/"
            "lerf_compat_20260520/logs/ and will run LEGaussians official "
            "quantize_features.py plus train.py after the LaGa phase in the robust "
            "follow-on chain. The jobs reuse the existing Mip-NeRF bicycle config as "
            "the schema/defaults and override LERF source/image/codebook paths on the "
            "CLI, so no generated LERF config is promoted yet. LEGaussians render/eval "
            "watchers are queued behind those follow-on chains and will run the "
            "official render_mask.py outputs through the same local LERF JSON-polygon "
            "evaluator before writing paper/artifacts/legaussians_lerf_summary."
        )
    if method == "LaGa" and submodule_missing is False:
        return (
            "Official implementation targets view-dependent semantics through object "
            "decomposition and descriptors. The kmeans_pytorch submodule is initialized "
            "through a local .gitmodules mapping; "
            "simple_knn and both diff rasterizers are built into "
            "output/baselines/laga/local_site; the PyTorch3D KNN dependency is bypassed "
            "by a chunked torch.cdist fallback; train_scene.py/train_affinity_features.py "
            "reach CLI help. paper/artifacts/laga_lerf_readiness_audit.{json,md} "
            "records that the four LERF scenes have data and labels but lack "
            "scene_point_cloud.ply, contrastive_feature_point_cloud.ply, and descriptor "
            "files from inference.ipynb. The robust GPU4/GPU5 follow-on chains are queued "
            "behind the Dr. Splat render/eval watchers under "
            "output/baselines/gpu_followon/lerf_compat_20260520/logs/. They require "
            "the Dr. Splat render/eval completion marker before starting, then GPU4 "
            "will run ramen then teatime and GPU5 will run figurines then "
            "waldo_kitchen. Each scene first restores the Occam 30k RGB checkpoint for "
            "one extra LaGa train_scene.py step to export scene_point_cloud.ply, then "
            "runs train_affinity_features.py for the contrastive feature point cloud. "
            "A 12-field Occam RGB checkpoint restore compatibility patch is in place "
            "so LaGa can initialize from RGB checkpoints that do not already contain "
            "its mask parameter. "
            "Strict comparison still needs those queued jobs to finish plus batch "
            "adaptation of inference.ipynb to export same-evaluator masks."
        )
    if method == "Semantic Gaussians" and submodule_missing is False:
        return (
            "Official implementation targets ScanNet/MVImgNet-style semantic projection. "
            "The local simple_knn/rgbd-rasterization/channel-rasterization/"
            "segment-anything local site is built under "
            "output/baselines/semantic_gaussians/local_site, with scikit-image, "
            "viser, TensorFlow, and NumPy pinned to a compatible import stack. "
            "The isolated official-stack env at "
            "output/baselines/semantic_gaussians/scannet_compat_20260520/envs/"
            "sega-py39-torch211-cu118 has Python 3.9.18, PyTorch 2.1.1, "
            "CUDA 11.8, and requirements.txt finished with "
            "requirements_after_minkowski_exit 0 in "
            "sega_requirements_after_minkowski_heartbeat_20260520.log. "
            "train.py/fusion.py/distill.py/eval_segmentation.py import successfully "
            "under the official env when loaded in the repository's entrypoint order. "
            "The host PyTorch 2.7.1/CUDA attempt still fails at "
            "MinkowskiEngine spmm.cu, but the isolated official-stack env at "
            "sega-py39-torch211-cu118 has MinkowskiEngine 0.5.4 now imports "
            "successfully after pointing CUDA_HOME at the conda CUDA toolkit. "
            "PyTorch-Encoding (`encoding`) now imports successfully alongside "
            "TensorFlow, viser, detectron2, CLIP, and the Semantic Gaussians "
            "entrypoints. OpenSeg SavedModel weights were downloaded from the "
            "Stanford mirror into /root/baselines/semantic-gaussians/weights/"
            "openseg_exported_clip for fusion.py. A 1-iteration RGB train smoke "
            "passed after a local external-repo compatibility patch for top-level "
            "Nerfstudio intrinsics in transforms_train.json; full Semantic RGB "
            "30k train chains are running on GPU4 (`scene0000_00` then "
            "`scene0070_00`) and GPU5 (`scene0062_00` then `scene0097_00`), with "
            "fusion watchers queued behind those completion markers and a "
            "label-PLY distill/eval watcher using eval_label_ply.py queued behind "
            "fusion because these extracted ScanNet scenes provide "
            "*_vh_clean_2.labels.ply rather than per-view label-filt PNGs. "
            "paper/artifacts/semantic_gaussians_readiness_audit.{json,md} records "
            "the current strict-ScanNet readiness state: core dependencies are "
            "importable from the official env, all four ScanNet scene zips are now "
            "extracted, all four scenes have usable raw language features, and "
            "RGB-GS/fusion/distill/eval outputs are absent. "
            "Same-protocol metric export is still pending."
        )
    if method == "CAGS" and submodule_missing is False:
        return (
            "OpenGaussian-compatible LERF training/render/eval scripts are available; "
            "the rasterizer ABI blocker and PyG source builds are cleared locally via "
            "output/baselines/cags/local_site, and train.py/render_lerf_by_text.py "
            "reach CLI help. The vectorized clustering fix, PyTorch checkpoint-loading "
            "wrapper, CPU-only FAISS fallback, and train/test render wrapper fix are "
            "in place. All four scenes completed local compatibility training/render/eval "
            "from OpenGaussian 30k starts: scene-mean mIoU 0.2627 / Acc@0.25 0.3997 "
            "and object-weighted mIoU 0.2394 / Acc@0.25 0.3558 over 208 objects, "
            "with 34 missing rendered masks counted. This is a diagnostic reproduced row, "
            "not a SOTA claim. ScanNet evaluation is marked TODO upstream."
        )
    return blocker


def _feature_stems(feature_dir: Path, suffix: str) -> set[str]:
    if not feature_dir.exists():
        return set()
    return {
        path.name.removesuffix(suffix)
        for path in feature_dir.glob(f"*{suffix}")
        if path.is_file()
    }


def inspect_repo(repo: Path, method: str, url: str, blocker: str) -> dict[str, Any]:
    exists = repo.exists()
    entry: dict[str, Any] = {
        "method": method,
        "url": url,
        "repo_path": str(repo),
        "exists": exists,
        "commit": None,
        "dirty": None,
        "submodule_missing": None,
        "submodule_status": [],
        "blocker": blocker,
    }
    if not exists:
        entry["blocker"] = f"Repository is not cloned locally. {blocker}"
        return entry

    entry["commit"] = _run_git(repo, ["rev-parse", "--short", "HEAD"]) or None
    status = _run_git(repo, ["status", "--short"])
    entry["dirty"] = bool(status)
    submodule_code, submodule_status = _run_git_result(repo, ["submodule", "status", "--recursive"])
    lines = [line for line in submodule_status.splitlines() if line]
    entry["submodule_status"] = lines
    entry["submodule_missing"] = (
        submodule_code != 0
        or any(line.startswith("-") or line.startswith("+") for line in lines)
    )
    entry["blocker"] = _blocker_for_repo_state(
        method,
        blocker,
        entry["submodule_missing"],
    )
    return entry


def inspect_lerf_assets(
    lerf_root: Path,
    scenes: Iterable[str] = LERF_SCENES,
) -> dict[str, dict[str, int | bool]]:
    assets: dict[str, dict[str, int | bool]] = {}
    for scene in scenes:
        scene_root = lerf_root / scene
        image_dir = scene_root / "images"
        direct_dir = scene_root / "language_features"
        langsplat_dir = scene_root / "langsplat" / "language_features"
        label_dir = lerf_root / "label" / scene
        image_stems = (
            {path.stem for path in image_dir.glob("*") if path.is_file()}
            if image_dir.exists()
            else set()
        )
        direct_mask_stems = _feature_stems(direct_dir, "_s.npy")
        direct_vector_stems = _feature_stems(direct_dir, "_f.npy")
        langsplat_mask_stems = _feature_stems(langsplat_dir, "_s.npy")
        langsplat_vector_stems = _feature_stems(langsplat_dir, "_f.npy")
        direct_complete = (
            bool(image_stems)
            and direct_mask_stems == image_stems
            and direct_vector_stems == image_stems
        )
        langsplat_complete = (
            bool(image_stems)
            and langsplat_mask_stems == image_stems
            and langsplat_vector_stems == image_stems
        )
        assets[scene] = {
            "scene_root_exists": scene_root.exists(),
            "images": len(image_stems),
            "labels": len(list(label_dir.rglob("*.jpg"))) if label_dir.exists() else 0,
            "direct_language_feature_masks": len(direct_mask_stems),
            "direct_language_feature_vectors": len(direct_vector_stems),
            "langsplat_language_feature_masks": len(langsplat_mask_stems),
            "langsplat_language_feature_vectors": len(langsplat_vector_stems),
            "direct_ready": direct_complete,
            "langsplat_complete_pairs": langsplat_complete,
        }
    return assets


def lerf_assets_ready_for_baselines(assets: dict[str, dict[str, int | bool]]) -> bool:
    return bool(assets) and all(bool(row.get("direct_ready")) for row in assets.values())


def inspect_occam_lerf_readout(
    occam_output_root: Path,
    scenes: Iterable[str] = LERF_SCENES,
) -> dict[str, Any]:
    """Inspect tracked OccamLGS pre-rendered LERF readout JSONs."""
    scene_rows: dict[str, dict[str, Any]] = {}
    weighted_loc_acc = 0.0
    weighted_miou = 0.0
    total_objects = 0
    complete = True
    for scene in scenes:
        path = occam_output_root / f"occamlgs_{scene}_lerf_prerendered_eval_script.json"
        row: dict[str, Any] = {
            "json": str(path),
            "exists": path.exists(),
            "loc_acc": None,
            "miou": None,
            "objects": 0,
        }
        if not path.exists():
            complete = False
            scene_rows[scene] = row
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            macro = payload["macro"]
            objects = int(macro["objects"])
            loc_acc = float(macro["loc_acc"])
            miou = float(macro["miou"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            complete = False
            scene_rows[scene] = row
            continue
        row.update({"loc_acc": loc_acc, "miou": miou, "objects": objects})
        scene_rows[scene] = row
        total_objects += objects
        weighted_loc_acc += loc_acc * objects
        weighted_miou += miou * objects

    if total_objects == 0:
        complete = False
        loc_acc_macro = None
        miou_macro = None
    else:
        loc_acc_macro = weighted_loc_acc / total_objects
        miou_macro = weighted_miou / total_objects

    return {
        "root": str(occam_output_root),
        "complete": complete,
        "objects": total_objects,
        "loc_acc": loc_acc_macro,
        "miou": miou_macro,
        "scenes": scene_rows,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _completed_lerf_mask_text(method: str, payload: dict[str, Any]) -> str | None:
    macro = payload.get("macro")
    if not isinstance(macro, dict):
        return None
    try:
        miou = float(macro["miou"])
        acc025 = float(macro["acc025"])
        acc05 = float(macro["acc05"])
        count = int(macro["count"])
        missing = int(macro.get("missing", 0))
    except (KeyError, TypeError, ValueError):
        return None
    return (
        f"{method} local LERF compatibility mask export/evaluation completed on all "
        f"four scenes with the shared nested-mask evaluator: mIoU {miou:.4f} / "
        f"Acc@0.25 {acc025:.4f} / Acc@0.5 {acc05:.4f} over {count} objects, "
        f"with {missing} missing masks counted. This is a same-evaluator local "
        "compatibility row; upstream released-checkpoint/protocol caveats remain "
        "separate from the metric export."
    )


def update_blockers_for_completed_reproductions(
    baselines: list[dict[str, Any]],
    artifact_root: Path = Path("paper/artifacts"),
    semantic_metrics_path: Path = Path(
        "output/baselines/semantic_gaussians/scannet_compat_20260520/"
        "semantic_gaussians_eval_metrics.json"
    ),
) -> list[dict[str, Any]]:
    """Promote blocker text from queued/pending to completed when summaries exist."""
    by_method = {str(row.get("method")): row for row in baselines}

    gags = _read_json(artifact_root / "gags_lerf_summary.json")
    if gags and isinstance(gags.get("scene_mean"), dict) and isinstance(gags.get("object_weighted"), dict):
        scene_mean = gags["scene_mean"]
        weighted = gags["object_weighted"]
        try:
            by_method["GAGS"]["blocker"] = (
                "Full local GAGS LERF compatibility training/eval completed on all "
                "four scenes from local feature extraction/training. Shared-summary "
                f"metrics: scene-mean LocAcc {float(scene_mean['locacc']):.4f} / "
                f"mIoU {float(scene_mean['miou']):.4f}; object-weighted LocAcc "
                f"{float(weighted['locacc']):.4f} / mIoU {float(weighted['miou']):.4f} "
                f"over {int(weighted['query_count'])} queries. This remains a local "
                "compatibility rerun because pretrained/preprocessed GAGS models are "
                "not released."
            )
        except (KeyError, TypeError, ValueError):
            pass

    drsplat = _read_json(artifact_root / "drsplat_lerf_summary.json")
    drsplat_text = _completed_lerf_mask_text("Dr. Splat", drsplat or {})
    if drsplat_text and "Dr. Splat" in by_method:
        by_method["Dr. Splat"]["blocker"] = (
            drsplat_text
            + " Official evaluation remains TBA upstream, so the local wrapper path is "
            "recorded explicitly."
        )

    legaussians = _read_json(artifact_root / "legaussians_lerf_summary.json")
    if (
        legaussians
        and isinstance(legaussians.get("scene_mean"), dict)
        and isinstance(legaussians.get("object_weighted"), dict)
        and "LEGaussians" in by_method
    ):
        scene_mean = legaussians["scene_mean"]
        weighted = legaussians["object_weighted"]
        try:
            by_method["LEGaussians"]["blocker"] = (
                "LEGaussians official quantize_features.py, train.py, and "
                "render_mask.py compatibility pipeline completed on all four LERF "
                f"scenes. Shared evaluator metrics: scene-mean mIoU "
                f"{float(scene_mean['miou']):.4f} / Acc@0.25 "
                f"{float(scene_mean['acc025']):.4f} / Acc@0.5 "
                f"{float(scene_mean['acc05']):.4f}; object-weighted mIoU "
                f"{float(weighted['miou']):.4f} over {int(weighted['count'])} "
                f"objects with {int(weighted['missing'])} missing masks counted. "
                "This is a local compatibility rerun, not a released-checkpoint macro."
            )
        except (KeyError, TypeError, ValueError):
            pass

    laga = _read_json(artifact_root / "laga_lerf_summary.json")
    laga_text = _completed_lerf_mask_text("LaGa", laga or {})
    if laga_text and "LaGa" in by_method:
        by_method["LaGa"]["blocker"] = (
            "LaGa scene point cloud export, affinity feature training, descriptor "
            "building, mask export, and shared evaluator pass completed. "
            + laga_text
            + " The row is reported as a compatibility adaptation of the inference "
            "notebook rather than an upstream paper-table macro."
        )

    semantic = _read_json(semantic_metrics_path)
    if semantic and isinstance(semantic.get("metrics"), dict) and "Semantic Gaussians" in by_method:
        try:
            mean_iou = float(semantic["metrics"]["mean_iou"])
        except (KeyError, TypeError, ValueError):
            mean_iou = None
        scenes = semantic.get("scenes")
        if mean_iou is not None and isinstance(scenes, dict):
            scene_parts = []
            for scene in sorted(scenes):
                try:
                    scene_parts.append(f"{scene} {float(scenes[scene]['miou']):.4f}")
                except (KeyError, TypeError, ValueError):
                    continue
            suffix = "; ".join(scene_parts)
            by_method["Semantic Gaussians"]["blocker"] = (
                "Semantic Gaussians ScanNet compatibility distill/eval completed on "
                "the four tracked ScanNet scenes using the local label-PLY evaluator. "
                f"Mean IoU is {mean_iou:.4f}"
                + (f" ({suffix}). " if suffix else ". ")
                + "This is a ScanNet-20 compatibility reproduction row; class-split "
                "leaderboard claims remain governed by the dedicated ScanNet protocol."
            )

    return baselines


def update_blockers_for_lerf_assets(
    baselines: list[dict[str, Any]],
    assets: dict[str, dict[str, int | bool]],
    occam_lerf_readout: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not lerf_assets_ready_for_baselines(assets):
        return baselines

    for row in baselines:
        if row.get("method") == "OpenGaussian":
            row["blocker"] = (
                "ScanNet is reproduced locally; four-scene LERF language_features are "
                "available, but a strict official-policy LERF training/evaluation rerun is pending."
            )
        elif row.get("method") == "OccamLGS":
            if occam_lerf_readout and occam_lerf_readout.get("complete"):
                row["blocker"] = (
                    "all four LERF compatibility scenes completed RGB training, "
                    "language-feature extraction, test feature-map rendering, and "
                    "tracked normalized pre-rendered readout: LocAcc "
                    f"{occam_lerf_readout['loc_acc']:.4f} / mIoU "
                    f"{occam_lerf_readout['miou']:.4f} over "
                    f"{occam_lerf_readout['objects']} objects. This remains a "
                    "compatibility readout, not a strict released-checkpoint macro."
                )
            else:
                row["blocker"] = (
                    "Four-scene LangSplat-format LERF language_features are available; "
                    "a clean build/train/eval pass is still pending."
                )
    return baselines


def link_complete_langsplat_features(
    lerf_root: Path,
    scenes: Iterable[str] = LERF_SCENES,
) -> dict[str, bool]:
    """Link LangSplat feature assets into the baseline-expected directory.

    OpenGaussian and OccamLGS read `scene/language_features`. The VALA helper
    writes the same `_s.npy`/`_f.npy` format under `scene/langsplat`.
    To avoid promoting partial smoke outputs, link only when every image stem
    has both feature files.
    """
    linked: dict[str, bool] = {}
    for scene in scenes:
        scene_root = lerf_root / scene
        image_dir = scene_root / "images"
        source_dir = scene_root / "langsplat" / "language_features"
        target_dir = scene_root / "language_features"
        image_stems = {path.stem for path in image_dir.glob("*") if path.is_file()}
        complete = bool(image_stems) and all(
            (source_dir / f"{stem}_s.npy").exists() and (source_dir / f"{stem}_f.npy").exists()
            for stem in image_stems
        )
        if not complete:
            linked[scene] = False
            continue
        if target_dir.exists() or target_dir.is_symlink():
            if target_dir.resolve() == source_dir.resolve():
                linked[scene] = True
                continue
            linked[scene] = False
            continue
        target_dir.symlink_to(source_dir, target_is_directory=True)
        linked[scene] = True
    return linked


def build_audit(
    baselines_root: Path,
    lerf_root: Path,
    occam_output_root: Path = Path("output/baselines/occamlgs/lerf_compat_20260518"),
    artifact_root: Path = Path("paper/artifacts"),
    semantic_metrics_path: Path = Path(
        "output/baselines/semantic_gaussians/scannet_compat_20260520/"
        "semantic_gaussians_eval_metrics.json"
    ),
) -> dict[str, Any]:
    lerf_assets = inspect_lerf_assets(lerf_root)
    occam_readout = inspect_occam_lerf_readout(occam_output_root)
    baselines = [
        inspect_repo(
            baselines_root / baseline["repo_dir"],
            baseline["method"],
            baseline["url"],
            baseline["blocker"],
        )
        for baseline in BASELINES
    ]
    baselines = update_blockers_for_lerf_assets(
        baselines,
        lerf_assets,
        occam_lerf_readout=occam_readout,
    )
    baselines = update_blockers_for_completed_reproductions(
        baselines,
        artifact_root=artifact_root,
        semantic_metrics_path=semantic_metrics_path,
    )
    return {
        "created": date.today().isoformat(),
        "baselines_root": str(baselines_root),
        "lerf_root": str(lerf_root),
        "baselines": baselines,
        "lerf_assets": lerf_assets,
        "occam_lerf_readout": occam_readout,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# External Baseline Audit",
        "",
        f"Created: `{payload.get('created', '-')}`",
        f"Baselines root: `{payload.get('baselines_root', '-')}`",
        f"LERF root: `{payload.get('lerf_root', '-')}`",
        "",
        "## Repositories",
        "",
        "| Method | Commit | Exists | Dirty | Missing submodule | Blocker |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("baselines", []):
        lines.append(
            "| {method} | {commit} | {exists} | {dirty} | {submodule_missing} | {blocker} |".format(
                method=row.get("method", "-"),
                commit=row.get("commit") or "-",
                exists=row.get("exists"),
                dirty=row.get("dirty"),
                submodule_missing=row.get("submodule_missing"),
                blocker=row.get("blocker", "-"),
            )
        )

    lines.extend(
        [
            "",
            "## LERF Language Features",
            "",
            "| Scene | Images | Labels | direct *_s/*_f | LangSplat *_s/*_f | Direct ready | LangSplat complete |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scene, row in payload.get("lerf_assets", {}).items():
        lines.append(
            "| {scene} | {images} | {labels} | {direct_s}/{direct_f} | {langsplat_s}/{langsplat_f} | {ready} | {langsplat_ready} |".format(
                scene=scene,
                images=row.get("images", 0),
                labels=row.get("labels", 0),
                direct_s=row.get("direct_language_feature_masks", 0),
                direct_f=row.get("direct_language_feature_vectors", 0),
                langsplat_s=row.get("langsplat_language_feature_masks", 0),
                langsplat_f=row.get("langsplat_language_feature_vectors", 0),
                ready=row.get("direct_ready", False),
                langsplat_ready=row.get("langsplat_complete_pairs", False),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_audit(payload: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines-root", type=Path, default=Path("/root/baselines"))
    parser.add_argument("--lerf-root", type=Path, default=Path("/mnt/pool/sqy/3d_understanding/lerf_ovs"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/baselines/external_baseline_audit/external_baseline_audit.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("output/baselines/external_baseline_audit/external_baseline_audit.md"),
    )
    parser.add_argument(
        "--occam-output-root",
        type=Path,
        default=Path("output/baselines/occamlgs/lerf_compat_20260518"),
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("paper/artifacts"))
    parser.add_argument(
        "--semantic-metrics",
        type=Path,
        default=Path(
            "output/baselines/semantic_gaussians/scannet_compat_20260520/"
            "semantic_gaussians_eval_metrics.json"
        ),
    )
    parser.add_argument(
        "--link-complete-langsplat",
        action="store_true",
        help="create scene/language_features symlinks only for complete LangSplat feature directories",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.link_complete_langsplat:
        link_complete_langsplat_features(args.lerf_root)
    payload = build_audit(
        args.baselines_root,
        args.lerf_root,
        args.occam_output_root,
        artifact_root=args.artifact_root,
        semantic_metrics_path=args.semantic_metrics,
    )
    write_audit(payload, args.output_json, args.output_md)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
