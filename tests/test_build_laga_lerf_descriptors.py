import numpy as np
import torch

from radio_gs.scripts import build_laga_lerf_descriptors as build


def test_multi_level_dimensions_match_laga_lerf_defaults():
    assert build.multi_level_dimensions(feature_dim=32, num_levels=3) == [16, 8, 8]
    assert build.multi_level_dimensions(feature_dim=32, num_levels=1) == [32]


def test_contiguous_cluster_labels_preserve_noise_as_minus_one():
    labels = np.asarray([-1, 7, 7, 3, -1, 3])

    remapped, unique = build.contiguous_cluster_labels(labels)

    assert unique == [3, 7]
    assert remapped.tolist() == [-1, 1, 1, 0, -1, 0]


def test_cluster_to_masks_excludes_noise_labels():
    labels = np.asarray([-1, 0, 1, 1, 0])

    masks = build.cluster_to_masks(labels, device=torch.device("cpu"))

    assert [mask.tolist() for mask in masks] == [
        [False, True, False, False, True],
        [False, False, True, True, False],
    ]


def test_filter_features_moves_mask_to_feature_device():
    features = torch.arange(12, dtype=torch.float32).view(4, 3)
    mask = torch.tensor([True, False, True, False])
    if torch.cuda.is_available():
        mask = mask.cuda()

    filtered = build.filter_features_with_mask(features, mask)

    assert filtered.device == features.device
    assert filtered.tolist() == [[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]]


def test_filter_features_handles_cluster_mask_after_prior_filtering():
    features = torch.arange(18, dtype=torch.float32).view(6, 3)
    first_mask = torch.tensor([True, True, False, True, False, False])
    second_mask = torch.tensor([False, True, True])
    if torch.cuda.is_available():
        first_mask = first_mask.cuda()
        second_mask = second_mask.cuda()

    filtered = build.filter_features_with_mask(features, first_mask)
    cluster_filtered = build.filter_features_with_mask(filtered, second_mask)

    assert cluster_filtered.device == features.device
    assert cluster_filtered.tolist() == [[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]]
