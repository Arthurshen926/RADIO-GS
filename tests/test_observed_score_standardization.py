import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField,standardize_observed_scores

def test_unknown_rows_do_not_change_map_moments_and_constant_scores_stay_finite():
    scores=torch.tensor([[1.,3.],[2.,3.],[3.,3.],[100.,-100.]])
    known=torch.tensor([True,True,True,False]);z=standardize_observed_scores(scores,known)
    assert torch.allclose(z[:3,0].mean(),torch.tensor(0.))
    assert torch.allclose(z[:3,0].std(unbiased=False),torch.tensor(1.))
    assert torch.equal(z[:3,1],torch.zeros(3))
    assert torch.isfinite(z).all()

def test_subset_and_query_batch_use_same_full_map_statistics():
    torch.manual_seed(4)
    f=ElementRetrievalField(15,4,{'text':7,'image':7},shared_query_groups=[['text','image']],calibrated_readout=True,native_siglip_readout=True,standardize_native_text=True)
    f.observed[:11]=True;q=torch.randn(3,7);rows=torch.tensor([1,3,5])
    assert torch.allclose(f.logits(q,'text')[rows],f.logits(q,'text',rows),atol=1e-5)
    assert torch.allclose(f.logits(q,'text')[:,:1],f.logits(q[:1],'text'),atol=1e-5)
    image=f.logits(q,'image');f.standardize_native_text=False
    assert torch.equal(image,f.logits(q,'image'))
