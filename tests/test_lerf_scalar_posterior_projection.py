import numpy as np
import pytest

from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    SelectionSpec,
    projection_semantics,
    scalar_posterior_mask,
    selected_membership_posterior_mask,
    validate_scalar_posterior_projection,
    validate_selected_membership_posterior_projection,
    visible_selected_membership_posterior,
)


def test_scalar_posterior_uses_unchanged_strict_score_threshold() -> None:
    values = np.array([[0.59, 0.60, 0.61]], dtype=np.float32)
    assert scalar_posterior_mask(values, 0.6).tolist() == [[False, False, True]]
    assert "continuous primitive query posterior" in projection_semantics(
        "scalar_posterior"
    )


def test_scalar_posterior_contract_rejects_hidden_selection_changes() -> None:
    valid = SelectionSpec("score_threshold", 0.6)
    validate_scalar_posterior_projection(
        "scalar_posterior",
        valid,
        selection_refinement="none",
        selection_min_ratio=0.0,
        selection_max_ratio=0.0,
    )
    with pytest.raises(ValueError, match="score_threshold"):
        validate_scalar_posterior_projection(
            "scalar_posterior",
            SelectionSpec("top_ratio", 0.1),
            selection_refinement="none",
            selection_min_ratio=0.0,
            selection_max_ratio=0.0,
        )
    with pytest.raises(ValueError, match="refinement"):
        validate_scalar_posterior_projection(
            "scalar_posterior",
            valid,
            selection_refinement="largest_component",
            selection_min_ratio=0.0,
            selection_max_ratio=0.0,
        )
    with pytest.raises(ValueError, match="ratio bounds"):
        validate_scalar_posterior_projection(
            "scalar_posterior",
            valid,
            selection_refinement="none",
            selection_min_ratio=0.01,
            selection_max_ratio=0.0,
        )


def test_scalar_posterior_rejects_nonfinite_maps() -> None:
    with pytest.raises(ValueError, match="finite 2D"):
        scalar_posterior_mask(np.array([[np.nan]], dtype=np.float32), 0.6)


def test_selected_membership_posterior_uses_fixed_bayes_majority() -> None:
    values = np.array([[0.49, 0.50, 0.51]], dtype=np.float32)
    alpha = np.ones_like(values)
    assert selected_membership_posterior_mask(values, alpha).tolist() == [
        [False, False, True]
    ]
    assert "visible selected-primitive membership" in projection_semantics(
        "selected_membership_posterior"
    )


def test_selected_membership_posterior_contract_is_fail_closed() -> None:
    valid = SelectionSpec("score_threshold", 0.6)
    validate_selected_membership_posterior_projection(
        "selected_membership_posterior",
        valid,
        selection_refinement="none",
        selection_min_ratio=0.0,
        selection_max_ratio=0.0,
        mask_refinement="none",
    )
    with pytest.raises(ValueError, match="score_threshold"):
        validate_selected_membership_posterior_projection(
            "selected_membership_posterior",
            SelectionSpec("top_ratio", 0.1),
            selection_refinement="none",
            selection_min_ratio=0.0,
            selection_max_ratio=0.0,
            mask_refinement="none",
        )
    with pytest.raises(ValueError, match="refinement"):
        validate_selected_membership_posterior_projection(
            "selected_membership_posterior",
            valid,
            selection_refinement="largest_component",
            selection_min_ratio=0.0,
            selection_max_ratio=0.0,
            mask_refinement="none",
        )
    with pytest.raises(ValueError, match="ratio bounds"):
        validate_selected_membership_posterior_projection(
            "selected_membership_posterior",
            valid,
            selection_refinement="none",
            selection_min_ratio=0.0,
            selection_max_ratio=0.1,
            mask_refinement="none",
        )
    with pytest.raises(ValueError, match="mask refinement"):
        validate_selected_membership_posterior_projection(
            "selected_membership_posterior",
            valid,
            selection_refinement="none",
            selection_min_ratio=0.0,
            selection_max_ratio=0.0,
            mask_refinement="largest_component",
        )


def test_selected_membership_posterior_rejects_nonfinite_maps() -> None:
    with pytest.raises(ValueError, match="finite 2D"):
        selected_membership_posterior_mask(
            np.array([[np.inf]], dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="shape-aligned"):
        selected_membership_posterior_mask(
            np.ones((1, 1), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        )


def test_selected_membership_posterior_rejects_low_alpha_halo() -> None:
    membership = np.ones((1, 2), dtype=np.float32)
    alpha = np.array([[0.001, 0.1]], dtype=np.float32)
    assert selected_membership_posterior_mask(membership, alpha).tolist() == [
        [False, True]
    ]


def test_selected_membership_posterior_uses_exact_png_round_support() -> None:
    membership = np.ones((1, 2), dtype=np.float32)
    alpha = np.array([[10.49 / 255.0, 10.51 / 255.0]], dtype=np.float32)
    assert selected_membership_posterior_mask(membership, alpha).tolist() == [
        [False, True]
    ]


def test_visible_membership_keeps_front_unselected_occlusion() -> None:
    # Front unselected opacity=.9; back selected opacity=.9 contributes only .1*.9.
    composite = np.array([[0.09]], dtype=np.float32)
    full_alpha = np.array([[0.99]], dtype=np.float32)
    posterior = visible_selected_membership_posterior(composite, full_alpha)
    assert posterior[0, 0] == pytest.approx(0.09 / 0.99)
    assert not selected_membership_posterior_mask(posterior, full_alpha)[0, 0]


def test_visible_membership_handles_empty_alpha_and_rejects_bad_shapes() -> None:
    posterior = visible_selected_membership_posterior(
        np.ones((2, 1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
    )
    assert posterior.tolist() == [[[0.0]], [[0.0]]]
    with pytest.raises(ValueError, match="spatial shapes"):
        visible_selected_membership_posterior(
            np.ones((2, 2), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
        )
