from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from radio_gs.querying.transient_rgb_sam import (
    PromptMode,
    TransientRgbSamPolicy,
    aggregate_sam_trials,
    calibrate_full_reference_interface,
    deterministic_signed_point_trials,
    observation_clamped_fusion,
    transient_adapter_contract,
)


def _small_policy(**overrides: object) -> TransientRgbSamPolicy:
    values = {
        "trials": 2,
        "positive_points_per_trial": 1,
        "negative_points_per_trial": 1,
        "prompt_pool_mass_fraction": 0.4,
        "maximum_prompt_pool_rows": 32,
        "signed_vote_threshold": 0.5,
        "sam_fusion_weight": 1.0,
        "full_mask_threshold_candidates": (0.25, 0.5, 0.75),
    }
    values.update(overrides)
    return TransientRgbSamPolicy(**values)


def test_signed_points_are_deterministic_bounded_and_separated() -> None:
    positive = np.full((4, 4), 0.2, dtype=np.float32)
    negative = np.full((4, 4), 0.2, dtype=np.float32)
    positive[0, 0] = 1.0
    negative[3, 3] = 1.0
    first = deterministic_signed_point_trials(
        positive, negative, image_shape=(8, 12), policy=_small_policy()
    )
    second = deterministic_signed_point_trials(
        positive, negative, image_shape=(8, 12), policy=_small_policy()
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    points, labels = first
    assert points.shape == (2, 2, 2)
    assert labels.tolist() == [[1, 0], [1, 0]]
    assert bool((points[..., 0] >= 0).all() and (points[..., 0] <= 11).all())
    assert bool((points[..., 1] >= 0).all() and (points[..., 1] <= 7).all())
    assert not bool((points[:, 0] == points[:, 1]).all(axis=-1).any())


def test_sam_trial_aggregation_keeps_candidate_axis() -> None:
    candidates = np.zeros((2, 3, 2, 2), dtype=np.float32)
    candidates[1, 2] = 1.0
    result = aggregate_sam_trials(candidates, policy=_small_policy())
    assert result.shape == (3, 2, 2)
    assert float(result[2].mean()) == pytest.approx(0.5)


def test_signed_points_reject_missing_exclusive_sign() -> None:
    prompt = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="exclusive positive and negative"):
        deterministic_signed_point_trials(
            prompt, prompt, image_shape=(4, 4), policy=_small_policy()
        )


def test_reference_calibration_is_reference_only_and_ties_are_stable() -> None:
    target = np.array([[1, 0], [0, 1]], dtype=bool)
    candidates = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.8, 0.2], [0.2, 0.8]],
        ],
        dtype=np.float32,
    )
    result = calibrate_full_reference_interface(
        candidates, target, policy=_small_policy()
    )
    assert result.branch == "sam"
    assert result.candidate_index == 0
    assert result.threshold == pytest.approx(0.25)
    assert result.reference_iou == pytest.approx(1.0)


def test_reference_calibration_can_choose_canonical_on_exact_tie() -> None:
    target = np.array([[1, 0], [0, 1]], dtype=bool)
    candidates = np.array([[[0.9, 0.1], [0.1, 0.9]]], dtype=np.float32)
    result = calibrate_full_reference_interface(
        candidates,
        target,
        canonical_probability=candidates[0],
        allow_canonical_fallback=True,
        policy=_small_policy(),
    )
    assert result.branch == "canonical"
    assert result.candidate_index == -1
    assert result.reference_iou == pytest.approx(1.0)


def test_observation_clamp_preserves_signed_evidence_and_conflicts() -> None:
    base = torch.tensor([[0.2, 0.3], [0.4, 0.5]])
    sam = torch.tensor([[0.9, 0.9], [0.9, 0.9]])
    positive = torch.tensor([[True, False], [True, False]])
    negative = torch.tensor([[False, True], [True, False]])
    fused, receipt = observation_clamped_fusion(
        base,
        sam,
        positive_observed=positive,
        negative_observed=negative,
        policy=_small_policy(),
    )
    torch.testing.assert_close(fused, torch.tensor([[1.0, 0.0], [0.4, 0.9]]))
    assert receipt["positive_exclusive"] == 1
    assert receipt["negative_exclusive"] == 1
    assert receipt["conflict"] == 1
    assert receipt["unknown"] == 1
    assert receipt["observed_values_preserved_exactly"] is True


def test_contract_makes_persistent_transient_boundary_explicit() -> None:
    signed = transient_adapter_contract(PromptMode.SIGNED_SCRIBBLE)
    full = transient_adapter_contract(PromptMode.FULL_REFERENCE_MASK)
    assert signed["persistent_scene_state"] is False
    assert signed["target_rgb_opened"] is True
    assert signed["target_mask_opened"] is False
    assert signed["full_reference_calibration_only"] is False
    assert full["full_reference_calibration_only"] is True


def test_lightweight_transient_import_does_not_load_scipy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from radio_gs.querying.transient_rgb_sam import FROZEN_POLICY; "
                "assert FROZEN_POLICY.trials == 10; "
                "assert 'scipy' not in sys.modules"
            ),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
