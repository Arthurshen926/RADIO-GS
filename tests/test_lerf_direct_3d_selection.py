import torch

from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    aggregate_scores_by_voxel,
    choose_registration_refiner,
    merge_registered_scores,
    refine_selection_by_voxel_components,
    score_text_aligned_embeddings,
    select_registration_frame_ids,
)


def test_select_registration_frame_ids_uses_official_available_frames():
    frames = select_registration_frame_ids(
        available_pose_ids=[1, 2, 3, 4, 5],
        annotated_frame_ids=[2, 3, 4],
        official_frame_ids=[3, 5, 9],
        mode="official",
        max_frames=0,
    )

    assert frames == [3]


def test_select_registration_frame_ids_evenly_subsamples_all_poses():
    frames = select_registration_frame_ids(
        available_pose_ids=[0, 2, 4, 6, 8],
        annotated_frame_ids=[2, 4],
        official_frame_ids=[],
        mode="all_poses",
        max_frames=3,
    )

    assert frames == [0, 4, 8]


def test_select_registration_frame_ids_supports_train_pose_subset():
    frames = select_registration_frame_ids(
        available_pose_ids=[0, 1, 2, 3, 4, 5],
        annotated_frame_ids=[1, 3],
        official_frame_ids=[],
        train_frame_ids=[0, 2, 4, 5],
        val_frame_ids=[1, 3],
        mode="train",
        max_frames=2,
    )

    assert frames == [0, 5]


def test_choose_registration_refiner_can_disable_vfa_for_ablation():
    refiner = object()

    assert choose_registration_refiner(refiner, disable_registered_refiner=False) is refiner
    assert choose_registration_refiner(refiner, disable_registered_refiner=True) is None


def test_score_text_aligned_embeddings_supports_canonical_relevancy():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text = torch.tensor([[1.0, 0.0]])
    canonical = torch.tensor([[0.0, 1.0]])

    scores = score_text_aligned_embeddings(
        embeddings,
        text,
        canonical_embeddings=canonical,
        scoring="relevancy",
        softmax_temperature=10.0,
    )

    assert scores.shape == (2, 1)
    assert scores[0, 0] > 0.99
    assert scores[1, 0] < 0.01


def test_merge_registered_scores_uses_fallback_for_unregistered_gaussians():
    registered = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    valid = torch.tensor([True, False])
    text = torch.tensor([[1.0, 0.0]])
    fallback = torch.tensor([[0.2], [0.7]])

    scores = merge_registered_scores(
        registered,
        valid,
        text,
        fallback_scores=fallback,
        scoring="cosine",
    )

    assert torch.allclose(scores, torch.tensor([[1.0], [0.7]]), atol=1e-6)


def test_aggregate_scores_by_voxel_dilate_propagates_neighbor_context():
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([[0.1], [0.9], [0.2]], dtype=torch.float32)

    same_voxel = aggregate_scores_by_voxel(
        scores,
        xyz,
        mode="voxel_max",
        resolution=6,
        blend=1.0,
    )
    dilated = aggregate_scores_by_voxel(
        scores,
        xyz,
        mode="voxel_max_dilate",
        resolution=6,
        blend=1.0,
    )

    assert same_voxel[0, 0] == scores[0, 0]
    assert dilated[0, 0] == scores[1, 0]


def test_refine_selection_by_voxel_components_keeps_top_score_components():
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.01, 1.0, 1.0],
            [1.02, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    selected = torch.ones(6, 1)
    scores = torch.tensor([[0.1], [0.2], [0.3], [0.9], [0.8], [0.7]])

    refined = refine_selection_by_voxel_components(
        selected,
        scores,
        xyz,
        mode="top_score_components",
        resolution=8,
        keep_components=1,
        min_component_size=1,
        rank_by="mean_score",
    )

    assert torch.equal(refined[:, 0], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))


def test_refine_selection_by_voxel_components_none_returns_input():
    xyz = torch.rand(4, 3)
    selected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    scores = torch.rand(4, 2)

    refined = refine_selection_by_voxel_components(
        selected,
        scores,
        xyz,
        mode="none",
        resolution=8,
        keep_components=1,
        min_component_size=1,
        rank_by="mean_score",
    )

    assert refined is selected
