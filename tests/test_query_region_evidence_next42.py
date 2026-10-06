import io
import torch
from radio_gs.querying.query_region_evidence import RegionEvidence,RelationHead,corrected_scores,normalize_membership,same_window


def fixture():
    v=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    bank=dict(indices=torch.tensor([[0,1,2],[0,1,2]]),weights=torch.ones(3),shape=[4,3],views=torch.tensor([0,0,1]),held_views=[],C=v,F=v,B=-v,tight=v,text=v,quality=torch.ones(3),area=torch.ones(3)*.1,disjoint=torch.tensor([[False,True,False],[True,False,False],[False,False,False]]))
    torch.manual_seed(0);head=RelationHead();return bank,head,RegionEvidence(bank,head)


def test_zero_and_saturated():
    _,_,r=fixture();q=torch.eye(3);p=torch.tensor([[-100.,100.,0.]]*4)
    assert torch.equal(r.logits(q,p),p)
    assert torch.equal(corrected_scores(r,q,p.sigmoid(),base_logits=p),p.sigmoid())


def test_unknown_unsupported_nontext():
    _,h,r=fixture();h.net[-1].bias.data.fill_(-1)
    q=torch.tensor([[1.,0.,0.]]);p=torch.zeros(4,1);z=r.logits(q,p)
    assert z[1]<0 and z[2]==0 and z[3]==0 and z[0]==0
    assert r.logits(q,p,'image') is p
    assert r.logits(q,p,'prompt') is p


def test_queries_batch_aba():
    _,h,r=fixture();h.net[-1].weight.data.fill_(.1);q=torch.eye(3);p=torch.zeros(4,3)
    a=r.logits(q[:1],p[:,:1]);r.logits(q[1:2],p[:,1:2]);assert torch.equal(a,r.logits(q[:1],p[:,:1]))
    assert torch.allclose(r.logits(q,p),torch.cat([r.logits(x[None],p[:,:1]) for x in q],1),atol=1e-7)


def test_view_exclusion_and_overlap():
    _,_,r=fixture();q=torch.eye(3);pos,neg,_,_,allowed=r.carriers(q,0)
    assert not pos[:2].any() and not neg[:2].any() and not allowed[:2].any()
    assert not r.carriers(q)[1][2].any()


def test_duplicate_mass():
    idx=torch.tensor([[0,0],[0,1]]);v=torch.ones(2);views=torch.zeros(2,dtype=torch.long)
    assert normalize_membership(idx,v,views,1).sum()==1
    idx=idx.repeat(1,3);assert normalize_membership(idx,v.repeat(3),views,1).sum().allclose(torch.tensor(1.))
    assert normalize_membership(idx,v.repeat(3)*.1,views,1).sum().allclose(torch.tensor(.1))


def test_gradients_and_cold_serialization():
    b,h,r=fixture();q=torch.eye(3);p=torch.zeros(4,3);loss=(r.logits(q,p)-1).square().mean();loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in h.parameters())
    assert h.net[-1].weight.grad.abs().sum()>0
    h.net[-1].bias.data.fill_(.2);buf=io.BytesIO();torch.save(dict(bank=b,state=h.state_dict()),buf);buf.seek(0);saved=torch.load(buf,weights_only=False);other=RelationHead();other.load_state_dict(saved['state']);cold=RegionEvidence(saved['bank'],other)
    assert torch.equal(r.logits(q,p),cold.logits(q,p))


def test_same_crop_intervention():
    rgb=torch.ones(3,20,20)*.8;mask=torch.zeros(20,20,dtype=torch.bool);mask[5:15,5:15]=True
    values,box=same_window(rgb,mask)
    assert values.shape==(3,3,384,384) and box==[2,2,18,18]
    assert torch.allclose(values[1]+values[2],values[0]+.5,atol=2e-6)
