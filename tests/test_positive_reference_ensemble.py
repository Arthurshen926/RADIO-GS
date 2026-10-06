import numpy as np
import pytest

from radio_gs.querying.positive_reference_ensemble import aggregate_positive_reference_trials


def fixture():
    positive=np.array([[True,True,False],[False,False,False]])
    parent=np.array([[True,True,False],[False,True,False]])
    return positive,parent


def test_weaker_full_scribble_coverage_rejected():
    positive,parent=fixture()
    weaker=parent.copy(); weaker[0,1]=False; weaker[1,2]=True
    mask,_,r=aggregate_positive_reference_trials(np.stack([parent,weaker]),positive,parent)
    assert r['accepted_trials']==[0]
    assert np.array_equal(mask,parent)


def test_identical_trials_reduce_parent():
    positive,parent=fixture()
    mask,_,r=aggregate_positive_reference_trials(np.stack([parent]*10),positive,parent)
    assert np.array_equal(mask,parent)
    assert not r['fallback_parent']


def test_none_admitted_falls_back_exactly():
    positive,parent=fixture()
    mask,_,r=aggregate_positive_reference_trials(np.zeros((10,*parent.shape),bool),positive,parent)
    assert np.array_equal(mask,parent) and r['fallback_parent']


def test_majority_ties_unknown_and_positive_retained():
    positive,parent=fixture()
    parent[0,1]=False
    left=parent.copy();right=parent.copy();left[1,2]=True
    mask,prob,r=aggregate_positive_reference_trials(np.stack([left,right]),positive,parent)
    assert prob[1,2]==.5 and not mask[1,2]
    assert mask[positive].all() and not r['sam_complement_is_negative']


def test_trial_permutation_invariant():
    positive,parent=fixture()
    trials=np.stack([parent]*3);trials[0,1,0]=True
    a,p,_=aggregate_positive_reference_trials(trials,positive,parent)
    b,q,_=aggregate_positive_reference_trials(trials[[2,0,1]],positive,parent)
    assert np.array_equal(a,b) and np.array_equal(p,q)


def test_inputs_immutable_and_bad_inputs_rejected():
    positive,parent=fixture(); trials=np.stack([parent]*10)
    old=[x.copy() for x in (trials,positive,parent)]
    aggregate_positive_reference_trials(trials,positive,parent)
    assert all(np.array_equal(x,y) for x,y in zip((trials,positive,parent),old))
    with pytest.raises(ValueError):
        aggregate_positive_reference_trials(trials.astype(float),positive,parent)
    with pytest.raises(ValueError):
        aggregate_positive_reference_trials(trials,np.zeros_like(positive),parent)
