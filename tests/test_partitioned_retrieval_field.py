import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.training.source_fusion_initialization import apply_source_fusion,fit_source_fusion


def test_structural_learning_cannot_change_native_readout_with_protected_prefix():
    torch.manual_seed(7)
    f=ElementRetrievalField(12,7,{'text':9,'image':9,'prompt':8},shared_query_groups=[['text','image']],native_siglip_readout=True,calibrated_readout=True,native_semantic_dim=4)
    f.observed[:]=True
    b=fit_source_fusion(torch.randn(7,9),torch.rand(7,12),f.observed,4)
    apply_source_fusion(f,b,freeze_decoder=True)
    original=f.latent[:,:4].detach().clone();q=torch.randn(3,9);before=f.logits(q,'text').detach().clone()
    mask=torch.ones_like(f.latent);mask[:,:4]=0;f.latent.register_hook(lambda g:g*mask)
    optimizer=torch.optim.Adam([f.latent,*f.query_adapters['prompt'].parameters()],lr=.03)
    prompt=torch.randn(2,8);target=torch.randn(12,2)
    structure=f.latent[:,4:].detach().clone()
    for _ in range(8):
        optimizer.zero_grad();loss=(f.logits(prompt,'prompt')-target).square().mean();loss.backward();optimizer.step()
    assert torch.equal(original,f.latent[:,:4])
    assert not torch.equal(structure,f.latent[:,4:])
    assert torch.equal(before,f.logits(q,'text'))
    other=ElementRetrievalField(12,7,{'text':9,'image':9,'prompt':8},shared_query_groups=[['text','image']],native_siglip_readout=True,calibrated_readout=True,native_semantic_dim=4)
    other.load_state_dict(f.state_dict());assert torch.equal(f.retrieve(q,'image')['scores'],other.retrieve(q,'image')['scores'])


def test_unprotected_semantic_update_changes_native_scores_at_same_capacity():
    torch.manual_seed(9)
    f=ElementRetrievalField(8,6,{'text':7,'image':7},shared_query_groups=[['text','image']],native_siglip_readout=True,native_semantic_dim=4)
    q=torch.randn(2,7);before=f.logits(q,'text').detach().clone()
    with torch.no_grad():f.latent[:,4:]+=100
    assert torch.equal(before,f.logits(q,'text'))
    with torch.no_grad():f.latent[:,:4]+=torch.randn(8,4)
    assert not torch.equal(before,f.logits(q,'text'))
