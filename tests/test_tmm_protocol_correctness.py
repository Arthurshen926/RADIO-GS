"""Hand-computable edge cases for independent protocol checks."""
import numpy as np
import pytest

from scripts.research.audit_tmm_protocol_correctness import (
    binary_counts, confusion_metrics, radius_pseudo_labels, threshold_grid_upper_bound)


def test_false_positives_and_false_negatives_are_both_in_union():
    got=binary_counts(np.array([1,1,0,0],bool),np.array([1,0,1,0],bool))
    assert (got['tp'],got['fp'],got['fn'])==(1,1,1)
    assert got['iou']==pytest.approx(1/3)
    assert got['precision']==got['recall']==.5


def test_void_prediction_is_error_but_out_of_split_gt_is_ignored():
    # Class 1: TP=1,FN=1; class 2: TP=1,FN=1; the last point is ignored.
    pred=np.array([1,0,1,2,2]);gt=np.array([1,1,2,2,40])
    got=confusion_metrics(pred,gt,[1,2,3])
    assert got['miou']==pytest.approx((1/3+1/2)/2)
    assert got['macc']==.5 and got['classes']==[1,2] and got['valid_count']==4


def test_weighted_confusion_uses_opacity_volume_mass():
    got=confusion_metrics(np.array([1,0,2]),np.array([1,1,2]),[1,2],np.array([1.,3.,2.]))
    assert got['miou']==got['macc']==.625


def test_class_balanced_density_is_not_majority_vote():
    # One class-1 point at centre beats three class-2 points further away.
    got,_=radius_pseudo_labels(np.zeros((1,3)),np.ones((1,3)),np.array([[1.,0,0,0]]),
        np.array([[0.,0,0],[.1,0,0],[.2,0,0],[.3,0,0]]),np.array([1,2,2,2]))
    assert got.tolist()==[1]


def test_threshold_oracle_is_marked_undeployable_and_handles_ties():
    got=threshold_grid_upper_bound(np.array([.2,.2,.1],np.float32),np.array([1,0,0],bool))
    assert got['iou']==.5
    assert got['oracle'] and not got['deployable']
