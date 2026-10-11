import numpy as np
import pytest
from scipy import sparse
from radio_gs.query_field.member_energy import solve_member_energy


def solve(a, prior, p, n=None, confidence=None, footprint=None, **kw):
    return solve_member_energy(sparse.csr_matrix(a), prior, np.array(p, bool),
        np.array(n if n is not None else [False]*len(p), bool),
        confidence if confidence is not None else np.ones(len(p)),
        footprint if footprint is not None else np.ones(len(prior)), cutoff=1., **kw)


def test_completes_missing_positive_and_preserves_untouched():
    selected, d = solve([[0, 2, 0]], [.9, .2, .8], [True])
    assert selected.tolist() == [True, True, True]
    assert d['added'] == 1


def test_removes_redundancy_without_negative_request():
    selected, d = solve([[2, 2]], [.9, .9], [True])
    assert selected.sum() == 1
    assert d['removed'] == 1


def test_retains_complementary_extent_when_all_factors_supplied():
    selected, _ = solve([[2, 0], [0, 2]], [.9, .9], [True, True])
    assert selected.all()


def test_reliability_changes_decision_not_just_metadata():
    high, _ = solve([[2]], [.2], [True], confidence=[1])
    low, _ = solve([[2]], [.2], [True], confidence=[.01])
    assert high[0] and not low[0]


def test_no_evidence_preserves_prior_even_with_footprint():
    for p, confidence in [([False], [1]), ([True], [0])]:
        selected, d = solve([[2, 2]], [.9, .2], p, confidence=confidence)
        assert selected.tolist() == [True, False]
        assert d['flips'] == 0


def test_missing_extent_is_not_guaranteed():
    selected, d = solve([[2, 2]], [.9, .9], [True])
    # An omitted second factor could uniquely require the deleted member.
    assert not selected.all() and not d['full_extent_guaranteed']


def test_energy_matches_independent_enumeration_and_is_monotone():
    rng = np.random.default_rng(7)
    for _ in range(30):
        a = rng.uniform(0, 2, (6, 5)); prior = rng.uniform(.1, .9, 5)
        p = np.array([1, 1, 1, 0, 0, 0], bool); n = ~p
        r = rng.uniform(0, 1, 6); f = rng.uniform(.1, 2, 5)
        selected, d = solve(a, prior, p, n, r, f, max_passes=100)
        def energy(s):
            covered = np.minimum(a @ s, 1)
            return ((r*p*(1-covered)/3)+(r*n*covered/3)).sum()+.1*(f@s)/(f@prior)+.1*(np.abs(2*prior-1)*(s != (prior>.5))).sum()/5
        assert d['energy_trace'][-1] == pytest.approx(energy(selected))
        assert np.max(np.diff(d['energy_trace'])) <= 1e-10
        assert d['converged']
        for g in range(5):
            neighbor = selected.copy(); neighbor[g] = ~neighbor[g]
            assert energy(neighbor) >= energy(selected)-1e-10
        for g in np.flatnonzero(selected):
            for h in np.flatnonzero(~selected):
                neighbor = selected.copy(); neighbor[g] = False; neighbor[h] = True
                assert energy(neighbor) >= energy(selected)-1e-10



def test_zero_prior_disables_undefined_footprint_ratio():
    selected, d = solve([[2]], [0.], [True])
    assert selected[0] and not d['footprint_enabled']


def test_rejects_conflicting_labels():
    with pytest.raises(ValueError, match='labels'):
        solve([[2]], [.9], [True], [True])


def test_local_solution_can_miss_better_member_exchange():
    # Expensive initial member and cheap alternative cover the same extent.
    # Neither single flip helps, while a simultaneous exchange would help.
    selected, d = solve([[2, 2]], [.9, .1], [True], footprint=[10, 1], exchange_candidates=0)
    assert selected.tolist() == [True, False]
    assert d['converged']
    alternative_energy = .1/9.1 + .1*.8
    assert alternative_energy < d['energy_trace'][-1]


def test_sparse_duplicate_entries_match_canonical_matrix():
    a = sparse.coo_matrix(([1., 1.], ([0, 0], [0, 0])), shape=(1, 1))
    result, details = solve(a, [.2], [True])
    expected, ref = solve([[2]], [.2], [True])
    assert np.array_equal(result, expected)
    assert details['energy_trace'] == ref['energy_trace']


def test_joint_exchange_escapes_verified_single_flip_trap():
    selected, d = solve([[2, 2]], [.9, .1], [True], footprint=[10, 1])
    assert selected.tolist() == [False, True]
    assert d['exchanges'] == 1 and d['converged']
    assert np.max(np.diff(d['energy_trace'])) <= 1e-10


def test_exchange_checks_entire_union_not_only_shared_positive():
    # Cheap member also covers a known negative: replacement must be rejected.
    selected, d = solve([[2, 2], [0, 2]], [.9, .1], [True, False],
                        [False, True], footprint=[10, 1])
    assert selected.tolist() == [True, False] and d['exchanges'] == 0


def test_negative_exchange_budget_rejected():
    with pytest.raises(ValueError, match='exchange'):
        solve([[2]], [.9], [True], exchange_candidates=-1)
