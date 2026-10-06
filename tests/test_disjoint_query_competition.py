import pytest
import torch
from radio_gs.training.disjoint_query_competition import disjoint_query_pairs, disjoint_query_order_loss, disjoint_query_order_accuracy


def test_containment_overlap_and_aliases_are_not_negative_relations():
    masks=torch.tensor([[1,1,0],[1,0,0],[0,0,1],[0,0,1]])
    descriptors=torch.eye(4)
    queries=descriptors.clone();queries[3]=queries[2]
    pairs=disjoint_query_pairs(masks,descriptors,queries).tolist()
    assert [0,1] not in pairs and [2,3] not in pairs
    assert [0,2] in pairs and [1,2] in pairs


def test_disjointness_without_reciprocal_teacher_preference_is_insufficient():
    masks=torch.eye(2)
    assert len(disjoint_query_pairs(masks,torch.tensor([[1.,0.],[1.,0.]]),torch.eye(2)))==0


def test_unknown_pixels_do_not_change_loss_or_receive_gradients():
    scores=torch.tensor([[.2,.8],[.7,.3],[.99,.01]],requires_grad=True)
    masks=torch.tensor([[1,0,0],[0,1,0]])
    loss=disjoint_query_order_loss(scores,masks,torch.ones(3,dtype=torch.bool))
    assert torch.allclose(loss,torch.tensor(.5))
    loss.backward()
    assert torch.equal(scores.grad[2],torch.zeros(2))
    assert scores.grad[0,0]<0 and scores.grad[0,1]>0


def test_correct_query_order_is_zero_loss_and_invalid_support_is_ignored():
    scores=torch.tensor([[.8,.2],[.3,.7]],requires_grad=True)
    assert disjoint_query_order_loss(scores,torch.eye(2),torch.ones(2,dtype=torch.bool))==0
    assert disjoint_query_order_loss(scores.flip(1),torch.eye(2),torch.zeros(2,dtype=torch.bool))==0


def test_overlap_is_rejected_in_actual_training_loss():
    with pytest.raises(ValueError,match='Overlapping'):
        disjoint_query_order_loss(torch.ones(2,2),torch.ones(2,2),torch.ones(2,dtype=torch.bool))


def test_one_sided_geometry_support_does_not_count_as_reciprocal_pair():
    scores=torch.tensor([[.1,.9],[.9,.1]],requires_grad=True)
    valid=torch.tensor([True,False])
    assert disjoint_query_order_accuracy(scores,torch.eye(2),valid) is None
    assert disjoint_query_order_loss(scores,torch.eye(2),valid)==0
