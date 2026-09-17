import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from scripts.audits.train_rendered_extent import balanced_mask_loss


def test_balanced_dice_output_is_not_calibrated_class_probability():
    # One positive and nine negatives with no separating features: at .5 the
    # balanced objective still pushes the common response UP, not to prior .1.
    response=torch.tensor(.5,requires_grad=True)
    target=torch.zeros(10,1);target[0]=1
    loss=balanced_mask_loss(response.expand(10,1),target,torch.ones(10,dtype=torch.bool))
    loss.backward();assert response.grad<0


def test_concept_bias_reduces_scope_before_clamp_and_preserves_other_queries():
    kwargs=dict(num_elements=8,latent_dim=6,query_dims={'text':4,'image':4},
        native_siglip_readout=True,native_semantic_dim=2,calibrated_readout=True,shared_query_groups=[['text','image']],
        grounded_transfer={'num_groups':1,'num_proposals':2,'identity_selection':{
            'metric':'support_iou','trusted_concept_start':1,'completion_mode':'native_within_extent'}})
    parent=ElementRetrievalField(**kwargs);parent.observed[:]=True
    with torch.no_grad():
        parent.grounded_transfer.acceptance.fill_(-2)
        parent.grounded_transfer.decoder.weight.zero_()
        parent.grounded_transfer.decoder.bias.copy_(torch.tensor([.2,.65]))
    child=ElementRetrievalField(**kwargs);child.load_state_dict(parent.state_dict())
    with torch.no_grad():child.grounded_transfer.decoder.bias[1:]-=.2
    q=torch.eye(4)[:2];before=parent.logits(q,'text').sigmoid();after=child.logits(q,'text').sigmoid()
    assert (before>.5).all() and (after<.5).all()
    assert torch.equal(parent.native_cosine(q),child.native_cosine(q))
    assert torch.equal(parent.logits(q,'image'),child.logits(q,'image'))
    assert torch.equal(parent.grounded_transfer.decoder.bias[:1],child.grounded_transfer.decoder.bias[:1])
    rows=torch.tensor([1,5]);torch.testing.assert_close(child.logits(q,'text',rows),child.logits(q,'text')[rows])
    clone=ElementRetrievalField(**kwargs);clone.load_state_dict(child.state_dict())
    assert torch.equal(clone.logits(q,'text'),child.logits(q,'text'))
    # Shifting the unclamped head is not subtracting from a probability afterward.
    raw=torch.tensor([1.4,-.3]);assert not torch.equal((raw-.2).clamp(0,1),(raw.clamp(0,1)-.2).clamp(0,1))
