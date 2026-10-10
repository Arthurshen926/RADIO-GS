import numpy as np
import torch
import pytest
from scipy import sparse
from types import SimpleNamespace
from radio_gs.query_field.joint_exclusion import _prune,_initial_weights,CUTOFF


def prune(matrix,positive,negative,score):
    matrix=sparse.csr_matrix(matrix);columns=matrix.tocsc();s=np.asarray(score);selected=s>.5
    return _prune(columns.indptr,columns.indices,columns.data,np.argsort(s,kind='stable'),selected,np.array(positive),np.array(negative),np.asarray(matrix@selected.astype(float)).ravel(),.5,.5001)[0]


def test_joint_redundancy_never_deletes_both_alternatives():
    # G0,G1 jointly explain P, but only G0 also leaks into N.
    assert prune([[1.,1.],[1.,0.]],[True,False],[False,True],[.6,.9]).tolist()==[False,True]
    # If BOTH leak into N, sequential protection retains one, not zero.
    assert prune([[1.,1.],[1.,1.]],[True,False],[False,True],[.6,.9]).tolist()==[False,True]


def test_complementary_positive_members_and_unknowns_are_retained():
    assert prune([[1.,0.],[0.,1.],[1.,1.]],[True,True,False],[False,False,True],[.6,.9]).all()
    assert prune([[1.,1.]],[False],[False],[.6,.9]).all()


def test_empty_and_already_satisfied_negatives_are_noop():
    assert not prune([[1.,1.]],[False],[True],[.1,.2]).any()
    assert prune([[.1,.1]],[False],[True],[.6,.9]).all()


def test_frozen_initial_visibility_underestimates_after_front_removal():
    ptr=np.array([0,2]);ids=np.array([0,1]);alpha=np.array([.8,.8]);selected=np.array([True,True])
    weights=_initial_weights(ptr,ids,alpha,selected)
    np.testing.assert_allclose(weights,[.8,.16])
    assert weights[1]<.8
