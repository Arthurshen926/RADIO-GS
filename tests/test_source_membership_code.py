import torch
from radio_gs.models.source_membership_code import SourceMembershipCode


def fixture():
    module=SourceMembershipCode(3,3,4)
    with torch.no_grad():
        module.decoder.weight.copy_(torch.eye(3))
        module.text_prototypes.copy_(torch.eye(4)[:3]);module.image_prototypes.copy_(torch.eye(4)[:3])
    structure=torch.tensor([[1.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    return module,structure,torch.ones(4,dtype=torch.bool)


def test_prototype_query_reads_membership_with_batch_row_invariance():
    module,z,observed=fixture();q=torch.eye(4)[:2];rows=torch.tensor([0,2,3])
    logits=module(q,'text',z,observed)
    assert torch.equal(logits.sigmoid()>.5,z[:,:2].bool())
    torch.testing.assert_close(module(q,'image',z,observed,rows=rows),logits[rows])
    torch.testing.assert_close(module(q[:1],'text',z,observed),logits[:,:1])
    restored=SourceMembershipCode(3,3,4);restored.load_state_dict(module.state_dict())
    assert torch.equal(restored(q,'text',z,observed),logits)


def test_signed_query_enforces_both_signs_and_unknown_abstention():
    module,z,observed=fixture()
    class Registration:
        def seed_rows(self,q):return q.long()
    query=torch.tensor([[0,2],[0,1],[3,0]])
    probability=module(query,'prompt',z,observed,Registration()).sigmoid()
    assert torch.equal(probability[:,0]>.5,z[:,0].bool())
    assert not (probability[:,1]>.5).any()
    assert torch.equal(probability[:,2]>.5,z[:,2].bool())
    observed[0]=False
    assert not (module(query,'prompt',z,observed,Registration()).sigmoid()>.5).any()


def test_group_balanced_factorization_restores_original_units():
    torch.manual_seed(3);matrix=torch.rand(4,20)*torch.tensor([1.,2.,10.,.1])[:,None]
    scale=matrix.norm(dim=1);normalized=matrix/scale[:,None]
    _,u=torch.linalg.eigh(normalized@normalized.T)
    code=normalized.T@u;decoder=u*scale[:,None]
    torch.testing.assert_close(code@decoder.T,matrix.T,atol=1e-5,rtol=1e-5)
