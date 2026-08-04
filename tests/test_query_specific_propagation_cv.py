import numpy as np
import pytest
import torch

from radio_gs.querying.query_specific_propagation_cv import (
    ACTION_HASH256_DIFFUSION,
    ACTION_STRONG_UNARY,
    audit_signed_cv_population,
    responsibility_balanced_log_loss,
    responsibility_weighted_auc,
    run_signed_scribble_cross_validation,
    stable_primitive_folds,
    training_evidence_for_fold,
)


def _manual_splitmix64(rows: np.ndarray) -> np.ndarray:
    values = rows.astype(np.uint64, copy=True)
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    values ^= values >> np.uint64(31)
    return (values % np.uint64(3)).astype(np.int64)


def test_stable_folds_are_global_row_deterministic_and_permutation_equivariant():
    rows = torch.tensor([0, 1, 2, 17, 1_000_003], dtype=torch.int64)
    expected = torch.from_numpy(_manual_splitmix64(rows.numpy()))
    torch.testing.assert_close(stable_primitive_folds(rows), expected)
    permutation = torch.tensor([4, 1, 3, 0, 2])
    torch.testing.assert_close(
        stable_primitive_folds(rows[permutation]), expected[permutation]
    )


def test_training_evidence_removes_only_heldout_fold():
    rows = torch.arange(30)
    folds = stable_primitive_folds(rows)
    evidence = torch.where(rows % 2 == 0, 0.5, -0.5)
    training = training_evidence_for_fold(evidence, folds, heldout_fold=1)
    assert bool((training[folds == 1] == 0).all())
    torch.testing.assert_close(training[folds != 1], evidence[folds != 1])


def test_population_audit_fails_closed_when_a_fold_lacks_a_class():
    rows = torch.arange(9)
    evidence = torch.ones(9)
    weights = torch.ones(9)
    with pytest.raises(ValueError, match="lacks the registered"):
        audit_signed_cv_population(
            rows, evidence, weights, minimum_class_rows=1
        )


def test_population_audit_reports_train_and_heldout_signed_counts():
    rows = torch.arange(600)
    evidence = torch.where(rows % 2 == 0, 0.4, -0.3)
    weights = torch.linspace(0.1, 2.0, 600)
    folds, reports = audit_signed_cv_population(
        rows, evidence, weights, minimum_class_rows=32
    )
    assert folds.shape == rows.shape
    assert len(reports) == 3
    for report in reports:
        assert report["heldout_positive_rows"] >= 32
        assert report["heldout_negative_rows"] >= 32
        assert report["training_positive_rows"] >= 32
        assert report["training_negative_rows"] >= 32


def test_responsibility_metrics_balance_classes_and_reward_perfect_ranking():
    labels = torch.tensor([True, True, False, False])
    probability = torch.tensor([0.9, 0.8, 0.2, 0.1])
    weight = torch.tensor([1.0, 9.0, 2.0, 8.0])
    loss = responsibility_balanced_log_loss(labels, probability, weight)
    auc = responsibility_weighted_auc(labels, probability, weight)
    assert 0 < loss < 0.3
    assert auc == pytest.approx(1.0)


def test_cross_validation_has_no_heldout_evidence_and_selects_better_unary():
    rows = torch.arange(600)
    evidence = torch.where(rows % 2 == 0, 0.5, -0.5)
    weights = torch.linspace(0.2, 2.0, 600)
    folds = stable_primitive_folds(rows)
    calls = []

    def predictor(training, fold):
        calls.append((training.clone(), fold))
        assert bool((training[folds == fold] == 0).all())
        labels = rows % 2 == 0
        unary = torch.where(labels, 0.9, 0.1)
        diffusion = torch.where(labels, 0.6, 0.4)
        return unary, diffusion

    result = run_signed_scribble_cross_validation(
        rows,
        evidence,
        weights,
        predictor,
        minimum_class_rows=32,
    )
    assert len(calls) == 3
    assert result.selected_action == ACTION_STRONG_UNARY
    assert (
        result.metrics[ACTION_STRONG_UNARY]["responsibility_balanced_log_loss"]
        < result.metrics[ACTION_HASH256_DIFFUSION][
            "responsibility_balanced_log_loss"
        ]
    )


def test_cross_validation_uses_auc_tiebreak_then_unary_complexity_tiebreak():
    rows = torch.arange(600)
    evidence = torch.where(rows % 2 == 0, 0.5, -0.5)
    weights = torch.ones(600)

    def identical_predictor(training, fold):
        del training, fold
        score = torch.full((600,), 0.5)
        return score, score.clone()

    result = run_signed_scribble_cross_validation(
        rows,
        evidence,
        weights,
        identical_predictor,
        minimum_class_rows=32,
    )
    assert result.selected_action == ACTION_STRONG_UNARY


def test_cross_validation_rejects_invalid_fold_probability():
    rows = torch.arange(600)
    evidence = torch.where(rows % 2 == 0, 0.5, -0.5)
    weights = torch.ones(600)

    def invalid_predictor(training, fold):
        del training, fold
        return torch.zeros(600), torch.full((600,), 1.2)

    with pytest.raises(ValueError, match="invalid probability"):
        run_signed_scribble_cross_validation(
            rows,
            evidence,
            weights,
            invalid_predictor,
            minimum_class_rows=32,
        )
