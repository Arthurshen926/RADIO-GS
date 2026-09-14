import pytest
import torch

from radio_gs.v4.carrier import SurfaceVoxelCarrier
from radio_gs.v4.evaluation.lerf_object_ceiling import (
    _canonical_membership,
    _evaluate_membership_ceiling_at_raster,
    _greedy_category_tokens,
)


def test_canonical_membership_uses_top2_simplex_and_preserves_unknown():
    dense, audit = _canonical_membership(torch.tensor([
        [0.8, 0.4, 0.0],
        [0.0, 0.0, 0.0],
    ]))
    assert dense.shape == (2, 3)
    assert dense[0].sum() == pytest.approx(1.0)
    assert dense[1].sum() == 0
    assert audit["raw_multi_token_element_fraction"] == pytest.approx(0.5)
    assert audit["canonical_unknown_mass_mean"] == pytest.approx(0.5)


def test_greedy_fixed_tokens_exposes_fragmentation_gain():
    rendered = {
        1: torch.tensor([[[True, False], [False, True]]]),
    }
    target = torch.tensor([[True, True]])
    single, single_score = _greedy_category_tokens(
        rendered, [(1, target)], maximum_tokens=1
    )
    top2, top2_score = _greedy_category_tokens(
        rendered, [(1, target)], maximum_tokens=2
    )
    assert len(single) == 1
    assert single_score == pytest.approx(0.5)
    assert sorted(top2) == [0, 1]
    assert top2_score == pytest.approx(1.0)


def test_ceiling_uses_one_stable_token_across_views():
    carrier = SurfaceVoxelCarrier(
        torch.tensor([[-0.5, 0.0, 2.0], [0.5, 0.0, 2.0]]),
        0.1,
        maximum_splat_radius=0,
        surface_band_voxels=0.0,
        maximum_contributors_per_pixel=1,
    )
    camera = type("View", (), {
        "intrinsic": torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        "camera_to_world": torch.eye(4),
        "width": 2,
        "height": 1,
    })()
    annotations = {
        1: [{"category": "object", "polygons": [torch.tensor([
            [0.0, 0.0], [0.49, 0.0], [0.49, 1.0], [0.0, 1.0]
        ]).numpy()]}],
        2: [{"category": "object", "polygons": [torch.tensor([
            [0.0, 0.0], [0.49, 0.0], [0.49, 1.0], [0.0, 1.0]
        ]).numpy()]}],
    }
    result = _evaluate_membership_ceiling_at_raster(
        carrier=carrier,
        membership=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        views={1: camera, 2: camera},
        annotations=annotations,
        categories=["object"],
        source_height=1,
        source_width=2,
        height=1,
        width=2,
        pixel_threshold=0.2,
    )
    assert result["query_selection_mode"] == "multi_instance"
    assert result["metrics"]["stable_category_single_token_miou"] == pytest.approx(1.0)
    assert result["metrics"]["leave_one_observation_out_single_token_miou"] == pytest.approx(1.0)


def test_raw_extent_diagnostic_is_explicitly_nondeployable():
    carrier = SurfaceVoxelCarrier(
        torch.tensor([[0.0, 0.0, 2.0]]),
        0.1,
        maximum_splat_radius=0,
        surface_band_voxels=0.0,
        maximum_contributors_per_pixel=1,
    )
    view = type("View", (), {
        "intrinsic": torch.eye(3),
        "camera_to_world": torch.eye(4),
        "width": 1,
        "height": 1,
    })()
    annotations = {1: [{"category": "object", "polygons": [torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]
    ]).numpy()]}]}
    result = _evaluate_membership_ceiling_at_raster(
        carrier=carrier,
        membership=torch.tensor([[0.8]]),
        views={1: view},
        annotations=annotations,
        categories=["object"],
        source_height=1,
        source_width=1,
        height=1,
        width=1,
        pixel_threshold=0.2,
        assignment_mode="raw_independent_token",
    )
    assert result["posterior_composition"].startswith("nondeployable_raw")
