import numpy as np
import pytest
from scipy import sparse

from radio_gs.querying.query_witness_anchor import QueryWitnessAnchor, weighted_coordinate_median


def test_equal_mass_is_coordinate_median_and_dominant_witness_moves_center():
    xyz = np.array([[0, 0, 0], [2, 1, 1], [5, 2, 3], [9, 4, 5]])
    np.testing.assert_allclose(weighted_coordinate_median(xyz, np.ones(4)), np.median(xyz, axis=0))
    np.testing.assert_allclose(weighted_coordinate_median(xyz, [0, 0, 1, 0]), xyz[2])
    with pytest.raises(ValueError): weighted_coordinate_median(xyz, [0, 0, 0, 0])


def test_other_view_evidence_moves_anchor_but_never_leaves_seed(monkeypatch):
    # First view's seed contains vertices 0,1,2; independent witnesses favor 2.
    m = sparse.csr_matrix([[1,0,0,0], [1,0,0,0], [1,1,1,0], [0,0,0,1]], dtype=float)
    v = sparse.csr_matrix(np.ones((4,3)))
    xyz = np.array([[0,0,0],[1,0,0],[2,0,0],[100,0,0]], dtype=float)
    reader = QueryWitnessAnchor(m,v,[0,1,2,2],xyz)
    monkeypatch.setattr('radio_gs.querying.query_witness_anchor.native_identity_posterior',
                        lambda values: (np.array([1.,1.,0.]), {'test':True}))
    anchor, d = reader.infer(0, np.zeros(4))
    assert anchor == 2 and d['source_views'] == 2 and d['seed_view_excluded']
    _, c = reader.infer(0, np.zeros(4), permute=True)
    assert c['source_weight_mass'] == d['source_weight_mass']
    assert c['normalized_view_mass'] == d['normalized_view_mass']
    assert c['admitted_regions'] == d['admitted_regions']


def test_missing_other_view_witness_falls_back_without_using_seed_as_proof():
    m = sparse.csr_matrix(np.ones((3,1)))
    reader = QueryWitnessAnchor(m, sparse.csr_matrix(np.ones((3,1))), [0],
                                np.array([[0,0,0],[1,0,0],[2,0,0]]))
    anchor, d = reader.infer(0, np.array([1.]))
    assert anchor == 1 and d['no_witness_fallback'] and d['source_views'] == 0


def test_all_regions_from_seed_view_are_excluded_before_query_gate(monkeypatch):
    seen=[]
    def gate(values):
        seen.append(values.copy())
        return np.ones(len(values)), {}
    monkeypatch.setattr('radio_gs.querying.query_witness_anchor.native_identity_posterior',gate)
    membership=sparse.csr_matrix([[1,1,0],[1,0,1],[0,1,1]],dtype=float)
    reader=QueryWitnessAnchor(membership,sparse.csr_matrix(np.ones((3,2))),[0,0,1],
        np.array([[0,0,0],[1,0,0],[2,0,0]]))
    first,_=reader.infer(0,np.array([100.,200.,.3]))
    second,_=reader.infer(0,np.array([-100.,-200.,.3]))
    assert first == second
    for values in seen: np.testing.assert_array_equal(values,[.3])
