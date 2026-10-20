import torch

from radio_gs.querying.response_permissions import admit_response_observation, exclusive_category_exclusion
from radio_gs.querying.response_refinement import refine_response, ResponseRefiner


def test_partial_region_has_no_background_authority():
    op = torch.eye(2).to_sparse()
    y = torch.tensor([[1.],[0.]])
    partial, complement = admit_response_observation(op,y,y>.5,y<.5,torch.tensor(1.))
    r0 = torch.tensor([[.3],[.8]])
    assert torch.equal(refine_response(r0,[partial])[0],torch.tensor([[1.],[.8]]))
    assert torch.equal(refine_response(r0,[complement])[0],y)


def test_reference_exclusion_only_within_supplied_domain():
    y = torch.tensor([[0.],[0.],[1.]])
    item,_ = admit_response_observation(torch.eye(3).to_sparse(),y,y>.5,y<.5,torch.tensor(1.),
        exclusion=torch.tensor([[True],[False],[False]]))
    result,_ = refine_response(torch.tensor([[.9],[.7],[.1]]),[item])
    assert torch.equal(result,torch.tensor([[0.],[.7],[1.]]))


def test_overlapping_or_missing_categories_supply_no_exclusion():
    foreground = torch.tensor([[True,False,False],[True,True,False],[False,False,False]])
    assert torch.equal(exclusive_category_exclusion(foreground),
        torch.tensor([[False,True,True],[False,False,False],[False,False,False]]))


def test_query_incompatible_evidence_cannot_modify_either_arm():
    y = torch.tensor([[1.],[0.]])
    pair = admit_response_observation(torch.eye(2).to_sparse(),y,y>.5,y<.5,torch.tensor(1.),
        compatible=torch.tensor(False))
    r0 = torch.tensor([[.2],[.8]])
    for item in pair:
        assert torch.equal(refine_response(r0,[item])[0],r0)


def test_streamed_controls_share_original_prior_without_changing_batch_rule():
    r0 = torch.tensor([[.3],[.8]])
    y = torch.tensor([[1.],[0.]])
    pair = admit_response_observation(torch.eye(2).to_sparse(),y,y>.5,y<.5,torch.tensor(1.))
    stream = ResponseRefiner(r0)
    for item in pair:
        stream.add(item)
    assert torch.equal(stream.finish()[0],refine_response(r0,pair)[0])
    assert torch.equal(r0,torch.tensor([[.3],[.8]]))


def test_eleven_nested_binary_masks_stay_within_probability_domain():
    import numpy as np
    from scipy import sparse
    from radio_gs.querying.response_study_adapters import mean_view_membership
    matrix=sparse.csr_matrix(np.ones((2,11),np.float32))
    y=mean_view_membership(matrix,np.arange(2),np.arange(11))
    assert y.dtype==np.float32
    assert np.array_equal(y,np.ones(2,np.float32))
