import argparse
import json

import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    COMPOSITOR_CONTRACT,
    PromptResponsibilityAuthority,
    PromptResponsibilityCache,
    save_prompt_responsibility_cache,
)
from radio_gs.querying.source_footprint_fold_authority import (
    load_source_footprint_fold_authority,
)
from radio_gs.scripts.build_source_footprint_fold_authority import build
from radio_gs.utils.immutable_artifacts import sha256_file


def test_build_source_footprint_authority_end_to_end_without_query_inputs(tmp_path):
    source_sha = "1" * 64
    authority = PromptResponsibilityAuthority(
        scene_id="scene",
        frame_id="frame",
        camera_name="frame",
        colmap_camera_name="00001",
        geometry_checkpoint_sha256="2" * 64,
        geometry_xyz_sha256="3" * 64,
        pose_sha256="4" * 64,
        intrinsics_sha256="5" * 64,
        height=8,
        width=8,
        num_gaussians=4,
        compositor_contract=COMPOSITOR_CONTRACT,
        source_sha256={"reference_binary_mask": source_sha},
    )
    cache = PromptResponsibilityCache(
        authority=authority,
        gaussian_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        pixel_ids=torch.tensor([0, 7, 63], dtype=torch.long),
        weights=torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32),
        visible_mass=torch.tensor([0.25, 0.5, 0.75, 0.0], dtype=torch.float64),
    )
    exact_path = tmp_path / "exact.pt"
    exact_artifact = save_prompt_responsibility_cache(cache, exact_path)
    exact_report = {
        "authority": authority.to_dict(),
        "authority_sha256": authority.digest,
        "file_sha256": exact_artifact.file_sha256,
        "tensor_bundle_sha256": cache.tensor_bundle_sha256,
        "historical_top1_responsibility_opened": False,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "reference_mask_header_authority": {
            "source_mask_pixels_decoded": False,
            "source_mask_pixels_interpreted": False,
            "query_or_evidence_constructed": False,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_computed": False,
        },
    }
    exact_report_path = tmp_path / "exact.json"
    exact_report_path.write_text(json.dumps(exact_report), encoding="utf-8")
    graph_path = tmp_path / "graph.pt"
    torch.save(
        {
            "global_rows": torch.tensor([0, 2, 3]),
            "num_global_rows": 4,
            "metadata": {
                "benchmark_images_opened": False,
                "benchmark_masks_opened": False,
                "text_queries_opened": False,
            },
        },
        graph_path,
    )
    output = tmp_path / "footprint.pt"
    report = build(
        argparse.Namespace(
            exact_w=str(exact_path),
            expected_exact_w_sha256=exact_artifact.file_sha256,
            exact_w_report=str(exact_report_path),
            expected_exact_w_report_sha256=sha256_file(exact_report_path),
            primitive_row_authority=str(graph_path),
            expected_primitive_row_authority_sha256=sha256_file(graph_path),
            output=str(output),
            report=None,
            overwrite=False,
        )
    )
    assert report["population"]["visible_rows"] == 2
    assert report["population"]["invisible_rows"] == 1
    assert report["primitive_row_authority"]["exact_w_row_filter"] == {
        "input_triplets": 3,
        "retained_triplets": 2,
        "excluded_triplets": 1,
        "input_weight_mass": 1.5,
        "retained_weight_mass": 1.0,
        "excluded_weight_mass": 0.5,
    }
    assert report["raster"]["block_count"] == 64
    assert report["source_mask_pixels_decoded"] is False
    loaded = load_source_footprint_fold_authority(
        output,
        expected_file_sha256=report["artifact"]["file_sha256"],
        expected_authority_sha256=report["artifact"]["authority_sha256"],
    )
    assert loaded.primitive_rows.tolist() == [0, 2, 3]
    assert loaded.group_ids.tolist() == [0, 63, 64]


def test_build_rejects_header_target_or_decode_flags(tmp_path):
    # The complete end-to-end test above covers tensor construction. This
    # small malformed report must fail before any exact-W or graph load.
    report_path = tmp_path / "bad.json"
    report_path.write_text(
        json.dumps(
            {
                "target_rgb_opened": False,
                "target_mask_opened": False,
                "reference_mask_header_authority": {
                    "source_mask_pixels_decoded": True,
                    "source_mask_pixels_interpreted": False,
                    "query_or_evidence_constructed": False,
                    "target_rgb_opened": False,
                    "target_mask_opened": False,
                    "target_metric_computed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        exact_w="missing",
        expected_exact_w_sha256="a" * 64,
        exact_w_report=str(report_path),
        expected_exact_w_report_sha256=sha256_file(report_path),
        primitive_row_authority="missing",
        expected_primitive_row_authority_sha256="b" * 64,
        output=str(tmp_path / "out.pt"),
        report=None,
        overwrite=False,
    )
    try:
        build(args)
    except ValueError as error:
        assert "source_mask_pixels_decoded" in str(error)
    else:
        raise AssertionError("decoded source mask flag must fail closed")
