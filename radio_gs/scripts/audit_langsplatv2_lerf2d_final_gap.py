#!/usr/bin/env python3
"""CPU-only final audit of the LangSplatV2 LERF-2D localization gap.

This audit intentionally does not render checkpoints.  It combines the exact-
camera cohort receipt with the raw annotations and the pinned upstream source
to determine which remaining differences can still be evaluation protocol
errors.  It also records why top-k/level/bbox counterfactuals cannot be replayed
without rendered relevance maps or a new GPU evaluation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


SCENE_ORDER = ("figurines", "teatime", "ramen", "waldo_kitchen")
SCENE_QUERY_COUNTS = {
    "figurines": 56,
    "teatime": 59,
    "ramen": 71,
    "waldo_kitchen": 22,
}
PAPER_2D_METHODS = (
    "LangSplat",
    "GAGS",
    "OccamLGS",
    "GOI",
    "GALA",
    "LangSplatV2",
)
PAPER_ROW_DECIMALS = {
    "LangSplat": 1,
    "GAGS": 2,
    "OccamLGS": 1,
    "GOI": 1,
    "GALA": 2,
    "LangSplatV2": 1,
}
PREDICTION_SUFFIXES = {
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".exr",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def _git_show(root: Path, path: str) -> str:
    return _git_output(root, "show", f"HEAD:{path}")


def _function_source(source: str, function_name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                segment = ast.get_source_segment(source, node)
                if segment is None:
                    break
                return segment
    raise ValueError(f"function {function_name!r} not found")


def _parse_paper_rows(table_path: Path) -> dict[str, dict[str, Any]]:
    text = table_path.read_text(encoding="utf-8")
    rows: dict[str, dict[str, Any]] = {}
    for method in PAPER_2D_METHODS:
        match = re.search(
            rf"&\s*{re.escape(method)}\s*&\s*([^\\]+?)\s*\\\\",
            text,
        )
        if match is None:
            raise ValueError(f"could not find 2D paper row for {method}")
        values = [float(item.strip()) for item in match.group(1).split("&")]
        if len(values) != 10:
            raise ValueError(f"{method}: expected 10 metric values, got {len(values)}")
        per_scene = {
            scene: {
                "miou_percent": values[2 + 2 * index],
                "loc_acc_percent": values[3 + 2 * index],
            }
            for index, scene in enumerate(SCENE_ORDER)
        }
        miou_macro = sum(item["miou_percent"] for item in per_scene.values()) / 4
        loc_macro = sum(item["loc_acc_percent"] for item in per_scene.values()) / 4
        denominator = sum(SCENE_QUERY_COUNTS.values())
        miou_micro = (
            sum(
                per_scene[scene]["miou_percent"] * SCENE_QUERY_COUNTS[scene]
                for scene in SCENE_ORDER
            )
            / denominator
        )
        loc_micro = (
            sum(
                per_scene[scene]["loc_acc_percent"] * SCENE_QUERY_COUNTS[scene]
                for scene in SCENE_ORDER
            )
            / denominator
        )
        decimals = PAPER_ROW_DECIMALS[method]
        tolerance = 0.5 * (10 ** (-decimals)) + 1e-12
        rows[method] = {
            "declared_mean_percent": {
                "miou": values[0],
                "loc_acc": values[1],
            },
            "per_scene_percent": per_scene,
            "recomputed": {
                "scene_macro_percent": {"miou": miou_macro, "loc_acc": loc_macro},
                "query_weighted_percent": {"miou": miou_micro, "loc_acc": loc_micro},
            },
            "numerical_match_at_printed_precision": {
                "miou_scene_macro": abs(values[0] - miou_macro) <= tolerance,
                "miou_query_weighted": abs(values[0] - miou_micro) <= tolerance,
                "loc_acc_scene_macro": abs(values[1] - loc_macro) <= tolerance,
                "loc_acc_query_weighted": abs(values[1] - loc_micro) <= tolerance,
            },
            "printed_decimals": decimals,
        }
    return rows


def _integer_hit_audit(query_count: int, printed_fraction: float) -> dict[str, Any]:
    # The source row is printed to one decimal percent, i.e. 0.001 in fraction.
    half_print_unit = 0.0005
    candidates = [
        hit
        for hit in range(query_count + 1)
        if abs(hit / query_count - printed_fraction) <= half_print_unit + 1e-12
    ]
    nearest = min(
        range(query_count + 1),
        key=lambda hit: abs(hit / query_count - printed_fraction),
    )
    return {
        "query_count": query_count,
        "printed_fraction": printed_fraction,
        "integer_hits_consistent_with_printed_precision": candidates,
        "nearest_integer_hits": nearest,
        "nearest_fraction": nearest / query_count,
        "nearest_delta_points": (nearest / query_count - printed_fraction) * 100,
    }


def _first_occurrence(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _scene_annotation_audit(
    scene: str,
    manifest: dict[str, Any],
    scene_metric: dict[str, Any],
) -> dict[str, Any]:
    query_count = 0
    raw_objects = 0
    duplicate_instances = 0
    bbox_count = 0
    bad_bboxes: list[dict[str, Any]] = []
    duplicate_categories: list[dict[str, Any]] = []
    role_query_counts = Counter({"train": 0, "test": 0})
    sizes: set[tuple[int, int]] = set()
    for frame, frame_manifest in manifest["frames"].items():
        annotation_path = Path(frame_manifest["annotation_path"])
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        info = payload["info"]
        if Path(str(info["name"])).stem != frame:
            raise ValueError(f"{annotation_path}: info.name does not match {frame}")
        width, height = int(info["width"]), int(info["height"])
        sizes.add((width, height))
        categories = [str(item["category"]) for item in payload["objects"]]
        queries = _first_occurrence(categories)
        if queries != list(frame_manifest["queries"]):
            raise ValueError(
                f"{annotation_path}: query order differs from evaluator order"
            )
        category_counts = Counter(categories)
        for category, count in category_counts.items():
            if count > 1:
                duplicate_categories.append(
                    {"frame": frame, "category": category, "instances": count}
                )
                duplicate_instances += count - 1
        for item in payload["objects"]:
            bbox = list(item["bbox"])
            bbox_count += 1
            if len(bbox) != 4:
                bad_bboxes.append({"frame": frame, "bbox": bbox, "reason": "not_xyxy"})
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            if (
                min(x1, x2) < 0
                or max(x1, x2) > width
                or min(y1, y2) < 0
                or max(y1, y2) > height
            ):
                bad_bboxes.append(
                    {"frame": frame, "bbox": bbox, "reason": "out_of_bounds"}
                )
        query_count += len(queries)
        raw_objects += len(payload["objects"])
        role_query_counts[str(frame_manifest["camera_role"])] += len(queries)

    if query_count != int(manifest["query_count"]):
        raise ValueError(f"{scene}: manifest query count mismatch")
    if query_count != int(scene_metric["objects"]):
        raise ValueError(f"{scene}: metric denominator mismatch")
    if bad_bboxes:
        raise ValueError(f"{scene}: invalid localization bboxes: {bad_bboxes}")

    paper_hits = _integer_hit_audit(
        query_count, float(scene_metric["paper"]["loc_acc"])
    )
    local_hits = int(scene_metric["loc_hits"])
    log_path = Path(scene_metric["log_path"])
    log_text = log_path.read_text(encoding="utf-8")
    levels_match = re.search(r"chosen_lvl:\s*\n?(\[[^]]*\])", log_text, re.DOTALL)
    if levels_match is None:
        raise ValueError(f"{log_path}: missing segmentation chosen_lvl list")
    segmentation_levels = ast.literal_eval(levels_match.group(1))
    if len(segmentation_levels) != query_count:
        raise ValueError(f"{log_path}: segmentation chosen_lvl count mismatch")

    return {
        "frames": int(manifest["frame_count"]),
        "queries": query_count,
        "raw_annotation_objects": raw_objects,
        "merged_duplicate_instances": duplicate_instances,
        "duplicate_categories": duplicate_categories,
        "localization_bboxes": bbox_count,
        "annotation_sizes_width_height": [list(size) for size in sorted(sizes)],
        "query_order_matches_released_first_occurrence_dict_order": True,
        "bbox_validation": "all xyxy boxes are within annotation dimensions",
        "query_counts_by_local_checkpoint_camera_role": dict(role_query_counts),
        "local_hits": local_hits,
        "local_loc_acc": local_hits / query_count,
        "paper_hit_audit": paper_hits,
        "nearest_paper_hit_deficit": paper_hits["nearest_integer_hits"] - local_hits,
        "segmentation_level_histogram": dict(Counter(segmentation_levels)),
        "segmentation_level_note": (
            "The released log serializes mIoU level choices only. Localization "
            "recomputes independent levels and does not serialize them."
        ),
    }


def _source_contract(baseline_root: Path) -> dict[str, Any]:
    eval_source = _git_show(baseline_root, "eval_lerf.py")
    localization = _function_source(eval_source, "localization_process_cuda")
    quick = _function_source(eval_source, "evaluate_quick")
    ground_truth = _function_source(eval_source, "eval_gt_lerfdata")
    openclip = _git_show(baseline_root, "eval/openclip_encoder.py")
    eval_shell = _git_show(baseline_root, "eval_lerf.sh")
    train_shell = _git_show(baseline_root, "train.sh")
    arguments = _git_show(baseline_root, "arguments/__init__.py")

    checks = {
        "localization_independent_of_mask_threshold": (
            "mask_thresh" not in localization and "thresh" not in localization
        ),
        "localization_uses_29x29_average_pool": (
            "scale = 29" in localization and "AvgPool2d" in localization
        ),
        "localization_selects_level_by_peak_argmax": "torch.argmax(score_lvl)"
        in localization,
        "localization_accepts_bbox_boundaries_inclusively": all(
            token in localization
            for token in (
                "cord_list[1] >= x_min",
                "cord_list[1] <= x_max",
                "cord_list[0] >= y_min",
                "cord_list[0] <= y_max",
            )
        ),
        "quick_evaluator_hardcodes_topk4": (
            "get_weights_and_indices(gaussians._language_feature_logits, 4)" in quick
        ),
        "quick_evaluator_does_not_read_cli_topk": "args.topk" not in quick,
        "released_shell_sets_topk4": "TOPK=4" in eval_shell,
        "released_shell_sets_mask_threshold_0p4": "--mask_thresh 0.4" in eval_shell,
        "queries_are_exact_annotation_dict_keys": (
            "clip_model.set_positives(list(img_ann.keys()))" in quick
        ),
        "repeated_categories_are_merged": (
            "np.concatenate" in ground_truth and "stack_mask" in ground_truth
        ),
        "openclip_is_vit_b16_laion2b": (
            'self.clip_model_type = "ViT-B-16"' in openclip
            and "laion2b_s34b_b88k" in openclip
        ),
        "openclip_negatives_match_release": (
            '("object", "things", "stuff", "texture")' in openclip
        ),
        "official_train_shell_omits_eval_flag": re.search(
            r"(^|\s)--eval(\s|$)", train_shell
        )
        is None,
        "modelparams_default_eval_false": "self.eval = False" in arguments,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"pinned upstream source-contract checks failed: {failed}")
    return {
        "pinned_revision": _git_output(baseline_root, "rev-parse", "HEAD").strip(),
        "checks": checks,
        "localization_contract": {
            "activation": "29x29 count_include_pad=False average pool",
            "level_selection": "argmax of each level's pooled peak",
            "point_selection": "all coordinates tied at the selected level's maximum",
            "hit_test": "any tied peak inside any merged instance bbox, inclusive boundary",
            "mask_threshold_dependency": False,
        },
        "query_contract": {
            "positive_prompts": "exact annotation category strings in first-occurrence order",
            "negative_prompts": ["object", "things", "stuff", "texture"],
            "text_encoder": "OpenCLIP ViT-B-16 laion2b_s34b_b88k",
        },
        "topk_contract": {
            "released_eval_shell": 4,
            "quick_evaluator_effective": 4,
            "cli_argument_effective_in_quick_path": False,
        },
        "official_training_split_contract": {
            "train_shell_passes_eval": False,
            "model_default_eval": False,
            "interpretation": "released train.sh trains the language field on all cameras",
        },
    }


def _prediction_inventory(log_root: Path, checkpoint_root: Path) -> dict[str, Any]:
    exact_run_candidates = []
    for current_root, directory_names, file_names in os.walk(log_root):
        directory_names[:] = [name for name in directory_names if name != "gt"]
        for file_name in file_names:
            path = Path(current_root) / file_name
            if path.suffix.lower() in PREDICTION_SUFFIXES:
                exact_run_candidates.append(str(path))

    checkpoint_candidates = []
    for scene in SCENE_ORDER:
        for level in (1, 2, 3):
            root = checkpoint_root / f"{scene}_0_{level}"
            # Render caches, when present in these compatibility outputs, live
            # at the level root or one named subdirectory. Avoid traversing the
            # large point-cloud tree during this CPU-only receipt check.
            candidates = list(root.glob("*")) + list(root.glob("*/*"))
            for path in candidates:
                if not path.is_file():
                    continue
                if path.name == "chkpnt10000.pth":
                    continue
                if path.suffix.lower() in PREDICTION_SUFFIXES:
                    checkpoint_candidates.append(str(path))
    return {
        "exact_run_prediction_or_tensor_candidates": sorted(exact_run_candidates),
        "checkpoint_side_render_cache_candidates": sorted(checkpoint_candidates),
        "replayable_localization_peaks_available": False,
        "per_query_hit_bits_available": False,
        "consequence": (
            "No CPU-only top-k, localization-level, tie, bbox, or camera-role hit "
            "counterfactual can be computed from this run. A rendered feature/relevance "
            "cache or a new evaluator pass is required."
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    cohort = json.loads(args.cohort_summary.read_text(encoding="utf-8"))
    paper_rows = _parse_paper_rows(args.paper_table)
    scene_audits: dict[str, dict[str, Any]] = {}
    for scene in SCENE_ORDER:
        manifest_path = Path(cohort["camera_manifests"][scene])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scene_audits[scene] = _scene_annotation_audit(
            scene,
            manifest,
            cohort["scene_metrics"][scene],
        )

    total_queries = sum(item["queries"] for item in scene_audits.values())
    total_local_hits = sum(item["local_hits"] for item in scene_audits.values())
    total_nearest_paper_hits = sum(
        item["paper_hit_audit"]["nearest_integer_hits"]
        for item in scene_audits.values()
    )
    role_queries = {
        role: sum(
            item["query_counts_by_local_checkpoint_camera_role"][role]
            for item in scene_audits.values()
        )
        for role in ("train", "test")
    }
    checkpoint_eval_values = {
        bool(level["eval"])
        for scene in cohort["checkpoint_cohorts"].values()
        for level in scene["levels"].values()
    }
    if checkpoint_eval_values != {True}:
        raise ValueError("local checkpoint cohort is not uniformly eval=True")

    source_contract = _source_contract(args.baseline_root)
    changed_paths = _git_output(args.baseline_root, "diff", "--name-only").splitlines()
    output_inventory = _prediction_inventory(args.log_root, args.checkpoint_root)
    if output_inventory["exact_run_prediction_or_tensor_candidates"]:
        raise ValueError("unexpected exact-run prediction cache appeared during audit")
    if output_inventory["checkpoint_side_render_cache_candidates"]:
        raise ValueError(
            "unexpected checkpoint-side render cache appeared during audit"
        )

    langsplatv2_aggregation = paper_rows["LangSplatV2"]
    neighbor_acc_macro_matches = {
        method: paper_rows[method]["numerical_match_at_printed_precision"][
            "loc_acc_scene_macro"
        ]
        for method in PAPER_2D_METHODS
        if method != "LangSplatV2"
    }
    result = {
        "schema_version": 1,
        "audit": "LangSplatV2 LERF-2D final CPU-only localization-gap audit",
        "inputs": {
            "cohort_summary": str(args.cohort_summary),
            "cohort_summary_sha256": _sha256(args.cohort_summary),
            "paper_table": str(args.paper_table),
            "paper_table_sha256": _sha256(args.paper_table),
            "baseline_root": str(args.baseline_root),
            "checkpoint_root": str(args.checkpoint_root),
            "log_root": str(args.log_root),
        },
        "paper_aggregation_audit": {
            "all_2d_rows": paper_rows,
            "langsplatv2_conclusion": (
                "The printed LangSplatV2 row numerically matches scene-macro mIoU "
                "and 208-query-weighted LocAcc. This is a row-specific numerical "
                "matching rule, not a benchmark-wide uniform aggregation rule."
            ),
            "langsplatv2": langsplatv2_aggregation,
            "neighboring_2d_acc_rows_match_scene_macro": neighbor_acc_macro_matches,
            "cross_row_caveat": (
                "Every other listed 2D baseline Acc mean matches its four-scene macro. "
                "The source-context table is therefore aggregation-heterogeneous."
            ),
        },
        "scene_audits": scene_audits,
        "totals": {
            "queries": total_queries,
            "raw_annotation_objects": sum(
                item["raw_annotation_objects"] for item in scene_audits.values()
            ),
            "merged_duplicate_instances": sum(
                item["merged_duplicate_instances"] for item in scene_audits.values()
            ),
            "localization_bboxes": sum(
                item["localization_bboxes"] for item in scene_audits.values()
            ),
            "query_counts_by_local_checkpoint_camera_role": role_queries,
            "local_hits": total_local_hits,
            "local_query_micro_loc_acc": total_local_hits / total_queries,
            "nearest_paper_hits": total_nearest_paper_hits,
            "nearest_paper_integer_micro_loc_acc": total_nearest_paper_hits
            / total_queries,
            "nearest_paper_hit_deficit": total_nearest_paper_hits - total_local_hits,
            "delta_vs_nearest_paper_integer_micro_points": (
                (total_local_hits - total_nearest_paper_hits) / total_queries * 100
            ),
            "delta_vs_printed_paper_overall_points": (
                total_local_hits / total_queries
                - float(cohort["paper_context"]["published_overall"]["loc_acc"])
            )
            * 100,
        },
        "released_source_contract": source_contract,
        "local_training_and_checkout_audit": {
            "all_twelve_local_checkpoints_eval_true": True,
            "local_eval_true_effect": (
                "LLFF holdout removes every eighth camera from language-field training"
            ),
            "queries_on_locally_withheld_labelled_views": role_queries["test"],
            "queries_on_locally_trained_labelled_views": role_queries["train"],
            "released_train_sh_effect": (
                "No --eval flag and ModelParams default eval=False; all cameras are used"
            ),
            "start_checkpoint_provenance": (
                "local Occam-compatible RGB checkpoints, not official LangSplatV2 assets"
            ),
            "language_feature_provenance": (
                "local compatibility features, not an official released checkpoint bundle"
            ),
            "dirty_checkout_paths": changed_paths,
            "dirty_checkout_impact": {
                "eval_lerf.py": (
                    "exact-camera selection plus disabled-by-default visualization plumbing; "
                    "localization scoring body is unchanged"
                ),
                "utils/loss_utils.py": (
                    "training-side float/epsilon compatibility edit; not imported by evaluator"
                ),
                "utils/vq_utils.py": (
                    "evaluation device-placement compatibility plus training-side explicit "
                    "float32 quantizer initialization"
                ),
            },
        },
        "existing_output_counterfactual_audit": output_inventory,
        "eligibility": {
            "complete_four_scene_cohort": True,
            "released_metric_code_intent": True,
            "strict_paper_reproduction": False,
            "paper_facing_role": "compatibility diagnostic only",
            "local_result_name": (
                "LangSplatV2 released-code-intent / local-checkpoint exact-camera diagnostic"
            ),
            "reasons_not_strict": [
                "local checkpoints use eval=True whereas released train.sh defaults to eval=False",
                "local checkpoints start from Occam-compatible RGB assets",
                "local compatibility language features and training patches are not paper assets",
                "official pretrained LangSplatV2 checkpoint identity is not established",
                "the source-context table is not a uniform same-protocol benchmark",
            ],
        },
        "diagnosis": {
            "remaining_gap_still_plausibly_caused_by_known_evaluator_protocol_error": False,
            "most_likely_class": "checkpoint/training/feature provenance mismatch",
            "support": [
                "query, GT merge, camera identity, metric denominator, prompts, top-k, level selection, bbox rule, and aggregation have been checked",
                "mask threshold cannot affect localization in released code",
                "the local and released training camera splits differ materially",
                "local mIoU is already 1.61 points above the printed row while LocAcc is about nine hits lower, which is not the signature of one remaining global aggregation or threshold bug",
            ],
            "causality_limit": (
                "The exact nine misses cannot be assigned to camera role, level, tie, or bbox "
                "without per-query localization peaks/hits or a new render."
            ),
        },
        "action": {
            "now": [
                "keep 61.51 mIoU / 79.8077 LocAcc as diagnostic-only",
                "do not tune mask threshold to close the LocAcc gap",
                "do not call query-micro LocAcc a benchmark-wide aggregation rule",
            ],
            "highest_information_strict_followup": (
                "Prefer the official pretrained checkpoint bundle. If unavailable, retrain from "
                "paper-equivalent RGB/features with clean pinned code and eval=False, then run the "
                "exact-camera evaluator while serializing per-query hit, selected localization "
                "level, peak coordinate(s), and camera role."
            ),
        },
    }
    if (
        total_queries != 208
        or total_local_hits != 166
        or total_nearest_paper_hits != 175
    ):
        raise ValueError("unexpected final-gap totals")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    log_root = root / "output/protocol_audit_20260731/langsplatv2_lerf2d_view_fix"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-summary",
        type=Path,
        default=log_root / "cohort_summary.json",
    )
    parser.add_argument(
        "--paper-table",
        type=Path,
        default=root / "paper/lerf_ovs_main_table.tex",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/root/baselines/LangSplatV2"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=root / "output/baselines/langsplatv2/lerf_compat_20260518",
    )
    parser.add_argument("--log-root", type=Path, default=log_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root / "output/protocol_audit_20260801/"
            "langsplatv2_lerf2d_final_gap_audit.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    totals = result["totals"]
    print(
        f"LocAcc {totals['local_hits']}/{totals['queries']}="
        f"{totals['local_query_micro_loc_acc']:.6f}; nearest paper reconstruction "
        f"{totals['nearest_paper_hits']}/{totals['queries']}="
        f"{totals['nearest_paper_integer_micro_loc_acc']:.6f}; "
        f"deficit={totals['nearest_paper_hit_deficit']} hits"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
