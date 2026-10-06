import pytest
import torch
from radio_gs.training.region_mean_consistent_evidence import region_mean_consistent_evidence


def test_disjoint_source_regions_unchanged_and_uncovered_unknown():
    masks = torch.tensor([[1,1,0,0,0],[0,0,1,1,0]],dtype=torch.bool)
    desc = torch.tensor([[1.,0.],[0.,1.]])
    result, report = region_mean_consistent_evidence(masks,desc)
    torch.testing.assert_close(result,torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.],[0.,0.]]))
    assert report['final_objective'] == 0 and report['covered_pixels'] == 4


def test_whole_part_mean_fit_improves_jointly_with_explicit_tradeoff():
    masks = torch.tensor([[1,1,1,1],[1,0,0,0]],dtype=torch.bool)
    desc = torch.eye(2)
    result, report = region_mean_consistent_evidence(masks,desc)
    initial = masks.float().T @ desc/masks.sum(0)[:,None]
    operator = masks.float()/masks.sum(1)[:,None]
    before = ((operator@initial-desc)**2).sum(1)
    after = ((operator@result-desc)**2).sum(1)
    # A convex joint objective does not guarantee every region individually.
    # The part improves while the already-good whole incurs a small tradeoff.
    assert after.sum()<before.sum() and after[1]<before[1]
    means=operator@result
    assert means[0,0]>means[0,1] and means[1,1]>means[1,0]
    assert report['final_objective'] < report['initial_region_mse']
    assert report['mean_pixel_change'] <= report['initial_region_mse']


def test_feature_rotation_and_region_order_equivariance():
    masks = torch.tensor([[1,1,1,1],[1,1,0,0],[0,1,1,0]],dtype=torch.bool)
    desc = torch.tensor([[.2,1.],[1.,.1],[.3,.7]])
    rotation = torch.tensor([[0.,1.],[-1.,0.]])
    result,_ = region_mean_consistent_evidence(masks,desc)
    permuted,_ = region_mean_consistent_evidence(masks.flip(0),desc.flip(0)@rotation)
    torch.testing.assert_close(permuted,result@rotation)


def test_near_duplicate_conflicting_masks_remain_finite_and_bounded():
    masks = torch.ones((2,100),dtype=torch.bool);masks[1,-1]=False
    result,report = region_mean_consistent_evidence(masks,torch.tensor([[1.,0.],[-1.,0.]]))
    assert torch.isfinite(result).all()
    assert report['final_objective'] <= report['initial_region_mse']


def test_empty_masks_fail_closed():
    with pytest.raises(ValueError,match='Empty source masks'):
        region_mean_consistent_evidence(torch.zeros(1,3),torch.ones(1,2))
