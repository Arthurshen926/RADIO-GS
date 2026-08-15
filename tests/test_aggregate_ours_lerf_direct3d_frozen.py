import json
from pathlib import Path

import pytest
import torch

from radio_gs.scripts.aggregate_ours_lerf_direct3d_frozen import (
    EXPECTED_FRAMES,
    EXPECTED_FREEZE_SHA256,
    EXPECTED_SCENES,
    FrozenDirect3DContract,
    FrozenDirect3DError,
    _tensor_sha256,
    aggregate_results_root,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import xyz_geometry_fingerprint
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, sha256_file


def _write(path: Path, value: bytes) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _contract(tmp_path: Path) -> FrozenDirect3DContract:
    return FrozenDirect3DContract(
        freeze_path=str(tmp_path / "freeze.yaml"),
        freeze_sha256=EXPECTED_FREEZE_SHA256,
        freeze_id="evaluation_protocols_20260801_v1",
        task_id="concept_lerf3d_vala",
        registry_row="lerf3d_vala_occam_geometry_exact_protocol_20260801",
        scenes=EXPECTED_SCENES,
        objects=208,
    )


def _build(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    shared_readout = _write(tmp_path / "readout.pth", b"readout")
    materializer = _write(tmp_path / "materializer.py", b"source")
    for index, scene in enumerate(EXPECTED_SCENES):
        category = f"object-{scene}"
        config = _write(tmp_path / f"{scene}.yaml", b"config")
        renderer = _write(tmp_path / f"{scene}.pth", b"renderer")
        descriptor = _write(tmp_path / f"{scene}-descriptor.pt", b"descriptor")
        text = _write(tmp_path / f"{scene}-text.pt", b"text")
        field = _write(tmp_path / f"{scene}-field.pth", b"field")
        xyz = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        valid = torch.tensor([True, False])
        scores = torch.tensor(
            [[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]],
            dtype=torch.float16,
        )
        geometry = xyz_geometry_fingerprint(xyz)
        sources = {
            "descriptor_cache": descriptor,
            "text_query_cache": text,
            "field_checkpoint": field,
            "readout_checkpoint": shared_readout,
            "renderer_geometry_checkpoint": renderer,
            "materializer_source": materializer,
        }
        authority = {
            "schema_version": 2,
            "artifact_type": "radio_gs_lerf_multiscale_primitive_query_score_cache",
            "contract": "radio_gs.lerf_multiscale_query_score_authority.v2",
            "score_semantics": "raw_independent_normalized_cosine",
            "score_formula": (
                "l2_normalize(descriptor) @ l2_normalize(text_embedding).T"
            ),
            "score_dtype": "torch.float16",
            "scale_axis": [
                {"id": "0.25", "value": 0.25, "unit": "meter"},
                {"id": "0.45", "value": 0.45, "unit": "meter"},
                {"id": "0.7", "value": 0.7, "unit": "meter"},
            ],
            "query_axis": {
                "ids": [category],
                "order_sha256": canonical_json_sha256([category]),
            },
            "geometry_axis": {
                "num_gaussians": 2,
                "xyz_sha256": geometry["xyz_sha256"],
                "renderer_xyz_sha256": geometry["xyz_sha256"],
                "valid_sha256": _tensor_sha256(valid),
                "field_checkpoint_sha256": field["sha256"],
                "readout_checkpoint_sha256": shared_readout["sha256"],
                "renderer_geometry_checkpoint_sha256": renderer["sha256"],
            },
            "query_scores_sha256": _tensor_sha256(scores),
            "source_artifacts": sources,
            "consumer_contracts": {
                "direct3d": {
                    "contract": "radio_gs.ours_lerf_direct3d_multiscale_query_scores.v2",
                    "tensor_layout": "[primitive_row,scale,query]",
                    "scale_selection": "downstream_frozen_VALA_readout_only",
                }
            },
            "calibration_constraints": {
                "softmax_applied": False,
                "temperature_applied": False,
                "peak_normalization_applied": False,
                "threshold_applied": False,
                "scale_reduction_applied": False,
                "benchmark_images_opened": False,
                "benchmark_annotations_opened": False,
                "benchmark_masks_opened": False,
                "benchmark_metrics_opened": False,
            },
        }
        cache_payload = {
            "version": 2,
            "contract": "radio_gs.ours_lerf_direct3d_multiscale_query_scores.v2",
            "query_scores": scores,
            "query_ids": [category],
            "scale_ids": ["0.25", "0.45", "0.7"],
            "scale_radii_m": [0.25, 0.45, 0.7],
            "xyz": xyz,
            "valid": valid,
            "geometry_fingerprint": geometry,
            "field_checkpoint_sha256": field["sha256"],
            "readout_checkpoint_sha256": shared_readout["sha256"],
            "renderer_geometry_checkpoint_sha256": renderer["sha256"],
            "authority": authority,
        }
        cache = tmp_path / "caches" / f"{scene}.pt"
        cache.parent.mkdir(exist_ok=True)
        torch.save(cache_payload, cache)
        cache_sha = sha256_file(cache)
        _json(
            cache.with_suffix(".pt.json"),
            {
                "status": "complete_calibration_free_query_score_materialization",
                "query_score_cache": {
                    "path": str(cache.resolve()),
                    "sha256": cache_sha,
                },
                "shared_renderer_authority": authority,
            },
        )
        metric = 0.2 + index * 0.1
        row = {
            "miou": metric,
            "acc025": metric + 0.1,
            "acc050": metric - 0.1,
            "n": 52,
            "selection_mode": "score_threshold",
            "selection_value": 0.6,
            "selection_tag": "thr0p6",
            "selection_refinement": "none",
            "mask_refinement": "none",
            "projection_mode": "selected_only_alpha",
            "per_category": {
                category: {
                    "miou": metric,
                    "acc025": metric + 0.1,
                    "acc050": metric - 0.1,
                    "n": 52,
                }
            },
        }
        checkpoint_contract = {
            "model_missing_keys": [],
            "model_unexpected_keys": [],
            "codec_missing_keys": [],
            "codec_unexpected_keys": [],
            "sharpener_missing_keys": [],
            "sharpener_unexpected_keys": [],
            "refiner_missing_keys": [],
            "refiner_unexpected_keys": [],
            "errors": [],
        }
        relative_cache = str(cache.relative_to(tmp_path))
        protocol = {
            "protocol_preset": "vala_repo_3d",
            "feature_source": "frozen Ours row-aligned three-scale primitive query-score cache",
            "feature_level_count": 3,
            "scale_ids": ["0.25", "0.45", "0.7"],
            "scale_radii_m": [0.25, 0.45, 0.7],
            "level_selection": "highest_raw_knn_smoothed_peak_per_query",
            "vala_knn_k": 10,
            "vala_repo_score_remap": "clip(2 * per_query_minmax - 1, 0, 1)",
            "vala_repo_effective_pre_remap_threshold": 0.8,
            "score_postprocess": "vala_knn_minmax",
            "selection_refinement": "none",
            "mask_refinement": "none",
            "proposal_smoothing": "none",
            "render_role": "render physically selected primitives only for mask evaluation",
            "projection_mode": "selected_only_alpha",
            "projection_semantics": "physically subset selected primitives and render their alpha",
            "alpha_binarization": "png_uint8_gt10",
            "silhouette_threshold": 10.0 / 255.0,
            "diagnostic_oracle_prompt": False,
            "geometry_alignment_maps": False,
            "score_aggregation": "none",
            "rgb_refinement_source": "",
            "registered_feature_cache": "",
            "score_cache": "",
            "metrics": ["mIoU", "Acc@0.25", "Acc@0.50", "boundary_f", "trimap_iou"],
            "checkpoint_contract": checkpoint_contract,
            "config_sha256": config["sha256"],
            "checkpoint_sha256": renderer["sha256"],
            "ours_multiscale_query_score_cache": relative_cache,
            "ours_multiscale_query_score_cache_sha256": cache_sha,
            "repo_commit": "a" * 40,
        }
        config_rel = str(Path(config["path"]).relative_to(tmp_path))
        checkpoint_rel = str(Path(renderer["path"]).relative_to(tmp_path))
        args = {
            "scene": scene,
            "config": config_rel,
            "checkpoint": checkpoint_rel,
            "ours_multiscale_query_score_cache": relative_cache,
            "protocol_preset": "vala_repo_3d",
            "score_source": "direct",
            "vala_knn_k": "10",
            "selection_mode": "score_threshold",
            "score_threshold": "0.6",
            "threshold_sweep": "",
            "mean_std_sweep": "",
            "ratio_sweep": "",
            "confidence_sweep": "",
            "selection_refinement": "none",
            "mask_refinement": "none",
            "proposal_smoothing": "none",
            "score_postprocess": "vala_knn_minmax",
            "projection_mode": "selected_only_alpha",
            "all_labeled_frames": "False",
            "registered_feature_cache": "",
            "score_cache": "",
            "external_query_score_cache": "",
            "external_query_feature_cache": "",
        }
        _json(
            root / scene / scene / "lerf_direct_3d_selection_results.json",
            {
                "args": args,
                "protocol": protocol,
                "scene": {
                    "scene": scene,
                    "config": config_rel,
                    "checkpoint": checkpoint_rel,
                    "official_frames": list(EXPECTED_FRAMES[scene]),
                    "official_frames_only": True,
                    "categories": [category],
                    "best_by_miou": "thr0p6",
                    "results": {"thr0p6": row},
                },
            },
        )
    return root


def test_aggregates_exact_four_scene_208_object_cohort(tmp_path):
    root = _build(tmp_path)
    result = aggregate_results_root(
        root, repo_root=tmp_path, contract=_contract(tmp_path)
    )
    assert result["status"] == "complete_exact_frozen_protocol_evaluation"
    assert result["cohort"] == {
        "scenes": list(EXPECTED_SCENES),
        "objects": 208,
        "labelled_frames": 22,
    }
    assert result["scene_equal_macro"]["miou"] == pytest.approx(0.35)
    assert result["scene_equal_macro"]["acc025"] == pytest.approx(0.45)
    assert result["scene_equal_macro"]["acc050"] == pytest.approx(0.25)


def test_rejects_sweep_or_extra_result_row(tmp_path):
    root = _build(tmp_path)
    path = root / "figurines" / "figurines" / "lerf_direct_3d_selection_results.json"
    payload = json.loads(path.read_text())
    payload["scene"]["results"]["thr0p5"] = payload["scene"]["results"]["thr0p6"]
    _json(path, payload)
    with pytest.raises(FrozenDirect3DError, match="sweep or extra"):
        aggregate_results_root(root, repo_root=tmp_path, contract=_contract(tmp_path))


def test_rejects_bound_config_sha_drift(tmp_path):
    root = _build(tmp_path)
    (tmp_path / "teatime.yaml").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        aggregate_results_root(root, repo_root=tmp_path, contract=_contract(tmp_path))
