import torch

from radio_gs.rendering.contribution_compositor import (
    build_compositing_variants,
    composite_feature_variants,
    contribution_rank,
    front_to_back_weights,
    gaussian_footprint_alphas,
    marginal_responsibility_statistics,
    primitive_visibility_purity,
)


def test_front_to_back_weights_reset_transmittance_per_pixel():
    pixel_ids = torch.tensor([1, 0, 1, 0])
    alphas = torch.tensor([0.5, 0.2, 0.5, 0.5])

    order, grouped_pixels, weights, accumulated = front_to_back_weights(
        pixel_ids, alphas, num_pixels=2
    )

    torch.testing.assert_close(grouped_pixels, torch.tensor([0, 0, 1, 1]))
    torch.testing.assert_close(alphas[order], torch.tensor([0.2, 0.5, 0.5, 0.5]))
    torch.testing.assert_close(weights, torch.tensor([0.2, 0.4, 0.5, 0.25]))
    torch.testing.assert_close(accumulated, torch.tensor([0.6, 0.75]))


def test_contribution_rank_is_descending_within_each_pixel():
    pixels = torch.tensor([0, 1, 0, 1, 0])
    weights = torch.tensor([0.2, 0.8, 0.9, 0.1, 0.4])

    rank = contribution_rank(pixels, weights)

    torch.testing.assert_close(rank, torch.tensor([2, 0, 0, 1, 1]))


def test_marginal_responsibility_is_continuous_and_parameter_free():
    pixels = torch.tensor([0, 0, 1])
    weights = torch.tensor([0.8, 0.2, 0.6])

    statistics = marginal_responsibility_statistics(
        pixels, weights, num_pixels=2
    )

    torch.testing.assert_close(
        statistics.responsibility, torch.tensor([0.8, 0.2, 1.0])
    )
    torch.testing.assert_close(
        statistics.target_weight, torch.tensor([0.64, 0.04, 0.6])
    )
    torch.testing.assert_close(statistics.pixel_mass, torch.tensor([1.0, 0.6]))
    torch.testing.assert_close(
        statistics.pixel_collision_purity, torch.tensor([0.68, 1.0])
    )


def test_primitive_visibility_purity_preserves_mass_and_marks_ambiguity():
    visible, pure, purity = primitive_visibility_purity(
        torch.tensor([0, 1, 0]),
        torch.tensor([0.8, 0.2, 0.6]),
        torch.tensor([0.64, 0.04, 0.6]),
        num_gaussians=3,
    )

    torch.testing.assert_close(visible, torch.tensor([1.4, 0.2, 0.0]))
    torch.testing.assert_close(pure, torch.tensor([1.24, 0.04, 0.0]))
    torch.testing.assert_close(
        purity, torch.tensor([1.24 / 1.4, 0.2, 0.0])
    )


def test_top1_and_mean_composite_expected_row_features():
    row_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    gids = torch.tensor([0, 1, 2])
    pixels = torch.tensor([0, 0, 0])
    weights = torch.tensor([0.6, 0.3, 0.1])
    variants = build_compositing_variants(
        pixels,
        weights,
        num_pixels=1,
        gammas=(),
        topk=(1,),
    )

    maps, mass = composite_feature_variants(
        row_features,
        gids,
        pixels,
        variants,
        height=1,
        width=1,
        channel_chunk_size=1,
        variant_chunk_size=2,
    )

    torch.testing.assert_close(maps["alpha_mean"][:, 0, 0], torch.tensor([0.7, 0.4]))
    torch.testing.assert_close(maps["top1"][:, 0, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(mass["alpha_mean"], torch.ones(1, 1))


def test_gamma_two_sharpens_mixture_without_changing_rows():
    row_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gids = torch.tensor([0, 1])
    pixels = torch.tensor([0, 0])
    weights = torch.tensor([0.8, 0.2])
    variants = build_compositing_variants(
        pixels,
        weights,
        num_pixels=1,
        gammas=(2.0,),
        topk=(),
    )

    maps, _ = composite_feature_variants(
        row_features,
        gids,
        pixels,
        variants,
        height=1,
        width=1,
    )

    torch.testing.assert_close(
        maps["alpha_mean"][:, 0, 0], torch.tensor([0.8, 0.2])
    )
    torch.testing.assert_close(
        maps["gamma_2"][:, 0, 0], torch.tensor([16.0 / 17.0, 1.0 / 17.0])
    )
    assert maps["gamma_2"][0, 0, 0] > maps["alpha_mean"][0, 0, 0]


def test_footprint_alpha_uses_half_pixel_centres():
    alpha = gaussian_footprint_alphas(
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([[[0.5, 0.5]]]),
        torch.tensor([[[2.0, 0.0, 2.0]]]),
        torch.tensor([[0.8]]),
        width=1,
    )
    torch.testing.assert_close(alpha, torch.tensor([0.8]))


def test_expected_depth_band_uses_rendered_reference_depth():
    variants = build_compositing_variants(
        torch.tensor([0, 0]),
        torch.tensor([0.6, 0.3]),
        num_pixels=1,
        depths=torch.tensor([1.0, 1.4]),
        reference_depth=torch.tensor([[1.35]]),
        gammas=(),
        topk=(),
        depth_tolerance=0.1,
        relative_depth_tolerance=0.0,
    )
    torch.testing.assert_close(
        variants["expected_depth_band"], torch.tensor([0.0, 0.3])
    )
