import torch
from radio_gs.rendering.modality_support import supported_modality_response


def test_different_evidence_domains_require_different_raster_denominators():
    image=torch.tensor([[[.48,.45,.6,.9],[.08,.45,.1,.9]]])
    alpha=torch.ones(1,2)
    first=supported_modality_response(image,alpha,query_index=0,modality_index=0,num_queries=1,num_modalities=2,shared_support=False)
    second=supported_modality_response(image,alpha,query_index=0,modality_index=1,num_queries=1,num_modalities=2,shared_support=False)
    assert torch.allclose(first,torch.tensor([[.8,0.]]))
    assert torch.allclose(second,torch.tensor([[.5,.5]]))


def test_shared_domain_matches_old_normalization():
    torch.manual_seed(2);raw=torch.rand(3,4,7);alpha=torch.ones(3,4);raw[...,-1]=.8
    for m in range(3):
        for q in range(2):
            actual=supported_modality_response(raw,alpha,query_index=q,modality_index=m,num_queries=2,num_modalities=3,shared_support=True)
            assert torch.equal(actual,raw[...,2*m+q]/raw[...,-1])
