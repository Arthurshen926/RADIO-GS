import argparse
import json
from pathlib import Path

import pytest
import torch
import yaml

from radio_gs.querying.multiview_region_memory import (
    aggregate_proposal_membership,
    method_contract,
    pool_region_token_set,
    project_anchor_to_feature_view,
    sample_native_mask_at_feature_centers,
    select_source_views,
)
from radio_gs.scripts.materialize_multiview_region_memory import (
    normalized_box_observation_domain,
    proposal_positive_mass,
)
from radio_gs.scripts.audit_nvos_multiview_region_memory_assets import run
from radio_gs.utils.immutable_artifacts import sha256_file


def _assignment(gaussian_ids, pixel_ids, weights):
    order = torch.argsort(torch.tensor(pixel_ids), stable=True)
    return {
        "gaussian_ids": torch.tensor(gaussian_ids, dtype=torch.int32)[order],
        "pixel_ids": torch.tensor(pixel_ids, dtype=torch.int32)[order],
        "weights": torch.tensor(weights, dtype=torch.float32)[order],
    }


def test_contract_is_prompt_generic_and_target_free():
    contract = method_contract()
    assert contract["accepted_reference_prompt_kinds"] == [
        "positive_negative_scribbles",
        "reference_binary_mask",
    ]
    assert contract["source_rgb_only"] is True
    assert contract["target_rgb_mask_or_metric_input"] is False
    assert contract["scene_specific_parameters"] is False
    assert contract["sam_box_policy"] == {
        "padding_pixels": 16,
        "resolution": 1008,
        "confidence_threshold": 0.0,
        "minimum_projected_anchor_overlap": 0.05,
        "candidate_tie_break": "official_score_then_candidate_index",
    }


def test_projection_keeps_probability_and_confidence_separate():
    assignment = _assignment(
        gaussian_ids=[0, 1, 2, 3],
        pixel_ids=[0, 0, 1, 3],
        weights=[0.8, 0.2, 0.5, 1.0],
    )
    projection = project_anchor_to_feature_view(
        torch.tensor([1.0, 0.0, 0.75, 0.0]),
        torch.tensor([1.0, 1.0, 0.5, 0.0]),
        assignment,
        height=2,
        width=2,
    )
    assert projection.probability.shape == (2, 2)
    assert projection.confidence.shape == (2, 2)
    assert projection.probability[0, 0] == pytest.approx(0.8)
    assert projection.confidence[0, 0] == pytest.approx(1.0 - torch.exp(torch.tensor(-1.0)).item())
    assert projection.probability[0, 1] == pytest.approx(0.75)
    assert projection.confidence[0, 1] == pytest.approx(1.0 - torch.exp(torch.tensor(-0.25)).item())
    assert projection.confidence[1, 1] == 0
    assert torch.equal(
        projection.seed,
        torch.tensor([[True, True], [False, False]]),
    )
    assert 0 < projection.positive_anchor_coverage <= 1
    assert 0 < projection.assignment_reliability <= 1


def test_view_selection_excludes_reference_and_forbidden_and_is_deterministic():
    q = torch.tensor([1.0, 0.0, 1.0])
    c = torch.ones(3)
    projections = [
        project_anchor_to_feature_view(
            q,
            c,
            _assignment([0, 1], [0, 1], [0.4, 0.4]),
            height=1,
            width=2,
        ),
        project_anchor_to_feature_view(
            q,
            c,
            _assignment([0, 2], [0, 1], [0.5, 0.5]),
            height=1,
            width=2,
        ),
        project_anchor_to_feature_view(
            q,
            c,
            _assignment([0, 2], [0, 1], [0.5, 0.5]),
            height=1,
            width=2,
        ),
        project_anchor_to_feature_view(
            q,
            c,
            _assignment([2], [0], [0.9]),
            height=1,
            width=2,
        ),
    ]
    selected = select_source_views(
        ["reference", "view_b", "view_a", "target"],
        projections,
        count=2,
        reference_frame_id="reference",
        forbidden_frame_ids=["target"],
    )
    assert [row.frame_id for row in selected] == ["view_a", "view_b"]
    assert all(row.frame_id not in {"reference", "target"} for row in selected)


def test_native_mask_sampling_uses_feature_cell_centers():
    native = torch.zeros((4, 8), dtype=torch.bool)
    native[1, 3] = True
    native[3, 7] = True
    sampled = sample_native_mask_at_feature_centers(
        native,
        feature_height=2,
        feature_width=4,
    )
    expected = torch.zeros((2, 4), dtype=torch.bool)
    expected[0, 1] = True
    expected[1, 3] = True
    assert torch.equal(sampled, expected)


def test_normalized_box_domain_and_positive_mass_are_box_local():
    domain = normalized_box_observation_domain(
        [0.5, 0.5, 0.5, 0.5],
        height=4,
        width=4,
    )
    assert torch.equal(
        domain,
        torch.tensor(
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ]
        ),
    )
    proposal = torch.ones((4, 4), dtype=torch.bool)
    assignment = _assignment(
        gaussian_ids=list(range(16)),
        pixel_ids=list(range(16)),
        weights=[1.0] * 16,
    )
    mass = proposal_positive_mass(
        assignment,
        proposal,
        domain,
        num_gaussians=16,
        view_reliability=0.5,
    )
    assert int((mass > 0).sum()) == 4
    assert float(mass.sum()) == pytest.approx(2.0)


def test_proposal_adjoint_is_box_local_and_hard_anchors_are_bitwise_overwritten():
    assignment = _assignment(
        gaussian_ids=[0, 1, 2, 3],
        pixel_ids=[0, 1, 2, 3],
        weights=[1.0, 0.5, 0.25, 1.0],
    )
    proposal = torch.tensor([[True, False], [True, True]])
    domain = torch.tensor([[True, True], [False, False]])
    anchor_q = torch.tensor([0.2, 0.3, 1.0, 0.0])
    anchor_c = torch.tensor([0.4, 0.5, 1.0, 1.0])
    hard = torch.tensor([False, False, True, True])
    result = aggregate_proposal_membership(
        [assignment],
        [proposal],
        [domain],
        [0.8],
        num_gaussians=4,
        anchor_probability=anchor_q,
        anchor_confidence=anchor_c,
        hard_anchor=hard,
    )
    assert result.probability[0] == 1
    assert result.probability[1] == 0
    assert result.confidence[0] == pytest.approx(1 - torch.exp(torch.tensor(-0.8)).item())
    assert result.confidence[1] == pytest.approx(1 - torch.exp(torch.tensor(-0.4)).item())
    assert torch.equal(result.probability[hard], anchor_q[hard])
    assert torch.equal(result.confidence[hard], anchor_c[hard])
    assert torch.equal(result.observed, torch.tensor([True, True, True, True]))


def test_region_tokens_are_one_normalized_token_per_source_view():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    assignments = [
        _assignment([0, 1], [0, 1], [1.0, 1.0]),
        _assignment([1, 2], [0, 1], [1.0, 0.5]),
    ]
    proposals = [
        torch.tensor([[True, False]]),
        torch.tensor([[True, True]]),
    ]
    tokens, reliability = pool_region_token_set(
        features,
        assignments,
        proposals,
        [0.8, 0.5],
    )
    assert tokens.shape == (2, 2)
    assert torch.allclose(torch.linalg.vector_norm(tokens, dim=1), torch.ones(2))
    assert torch.equal(reliability, torch.tensor([0.8, 0.5]))


def test_invalid_assignment_or_incomplete_anchor_overwrite_fails_closed():
    with pytest.raises(ValueError, match="pixel order"):
        project_anchor_to_feature_view(
            torch.ones(2),
            torch.ones(2),
            {
                "gaussian_ids": torch.tensor([0, 1]),
                "pixel_ids": torch.tensor([1, 0]),
                "weights": torch.ones(2),
            },
            height=1,
            width=2,
        )
    with pytest.raises(ValueError, match="anchor overwrite"):
        aggregate_proposal_membership(
            [_assignment([0], [0], [1.0])],
            [torch.ones((1, 1), dtype=torch.bool)],
            [torch.ones((1, 1), dtype=torch.bool)],
            [1.0],
            num_gaussians=1,
            anchor_probability=torch.ones(1),
        )


def _source_inventory_fixture(tmp_path):
    rgb_root = tmp_path / "images"
    rgb_root.mkdir()
    source_names = ["frame_a.jpg", "frame_b.jpg", "frame_c.jpg"]
    for index, name in enumerate(source_names):
        (rgb_root / name).write_bytes(bytes([index + 1]) * (index + 3))
    target = "frame_target.jpg"

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    frame_order = ["frame_c.jpg", "frame_a.jpg", "frame_b.jpg"]
    frame_manifest = {
        "scene": "scene",
        "image_sort_mode": "numeric_then_exact_filename",
        "frame_id_mode": "source_rank",
        "excluded_image_names": [target],
        "num_frames": 3,
        "features": {"backbone": {"grid": [1, 2]}},
        "frames": [
            {
                "source_rank": index,
                "frame_idx": index,
                "source_file": name,
                "saved_stem": f"rgb_{index}",
            }
            for index, name in enumerate(frame_order)
        ],
    }
    (feature_dir / "frame_manifest.json").write_text(json.dumps(frame_manifest))
    train_ids = tmp_path / "train_frame_ids.json"
    train_ids.write_text(json.dumps({"frame_ids": [0, 1, 2]}))
    camera_map = tmp_path / "camera_map.json"
    camera_map.write_text(
        json.dumps(
            {
                "complete_colmap_coverage": True,
                "records": [
                    {"rgb_path": str(rgb_root / name)}
                    for name in source_names
                ]
                + [{"rgb_path": str(rgb_root / target)}],
            }
        )
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "feature_dir": str(feature_dir),
                "train_frame_ids_path": str(train_ids),
                "camera_map_path": str(camera_map),
            }
        )
    )
    responsibility = tmp_path / "responsibility.pt"
    torch.save(
        {
            "schema_version": 1,
            "metadata": {
                "schema_version": 1,
                "assignment_mode": "raster_gaussian_top1",
                "registration_weight_mode": "alpha_depth",
                "config": str(config),
                "checkpoint": str(tmp_path / "checkpoint.pth"),
                "selected_dataset_indices": [0, 1, 2],
                "selected_frame_indices": [0, 1, 2],
                "feature_height": 1,
                "feature_width": 2,
                "pose_sha256": "a" * 64,
                "intrinsics_sha256": "b" * 64,
                "xyz_sha256": "c" * 64,
                "gaussian_state_sha256": "d" * 64,
                "benchmark_images_opened": False,
                "benchmark_masks_opened": False,
                "text_queries_opened": False,
            },
            "assignments": [
                _assignment([0], [0], [0.5]),
                _assignment([1], [0], [0.5]),
                _assignment([2], [1], [0.5]),
            ],
        },
        responsibility,
    )
    manifest = {
        "schema_version": 1,
        "benchmark": "nvos",
        "protocol": {
            "target_rgb_at_query": "forbidden",
            "target_mask_use": "scoring_only",
        },
        "scenes": [
            {
                "scene_id": "scene",
                "target_rgb_policy": "excluded_from_field_training_and_query",
                "prompt": {
                    "type": "reference_binary_mask",
                    "frame_id": "frame_c",
                },
                "prompt_frame_ids": ["frame_c"],
                "evaluation_frame_ids": ["frame_target"],
                "excluded_training_frame_ids": ["frame_target"],
                "frames": [
                    {
                        "frame_id": "frame_target",
                        "rgb_path": str(rgb_root / target),
                    }
                ],
                "rgb_directory": str(rgb_root),
                # Deliberately differs from assignment/frame-manifest order.
                "training_frames": [
                    {
                        "frame_id": Path(name).stem,
                        "rgb_path": str(rgb_root / name),
                    }
                    for name in source_names
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, responsibility, feature_dir / "frame_manifest.json"


def test_asset_audit_uses_frame_manifest_assignment_order_and_never_opens_target(tmp_path):
    manifest, responsibility, _ = _source_inventory_fixture(tmp_path)
    output = tmp_path / "inventory.json"
    report = run(
        argparse.Namespace(
            manifest=str(manifest),
            manifest_sha256=sha256_file(manifest),
            scene_binding=[
                ["scene", str(responsibility), sha256_file(responsibility)]
            ],
            output=str(output),
        )
    )
    payload = json.loads(output.read_text())
    scene = payload["scenes"]["scene"]
    assert report["source_views"] == {"scene": 3}
    assert [row["source_file"] for row in scene["source_views"]] == [
        "frame_c.jpg",
        "frame_a.jpg",
        "frame_b.jpg",
    ]
    assert scene["reference_assignment_view_index"] == 0
    assert scene["safety"]["target_rgb_content_opened_or_hashed"] is False
    assert not (tmp_path / "images" / "frame_target.jpg").exists()


def test_asset_audit_fails_closed_if_frame_manifest_source_set_contains_target(tmp_path):
    manifest, responsibility, frame_manifest = _source_inventory_fixture(tmp_path)
    payload = json.loads(frame_manifest.read_text())
    payload["frames"][2]["source_file"] = "frame_target.jpg"
    frame_manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source set differs"):
        run(
            argparse.Namespace(
                manifest=str(manifest),
                manifest_sha256=sha256_file(manifest),
                scene_binding=[
                    ["scene", str(responsibility), sha256_file(responsibility)]
                ],
                output=str(tmp_path / "blocked.json"),
            )
        )
    assert not (tmp_path / "images" / "frame_target.jpg").exists()
