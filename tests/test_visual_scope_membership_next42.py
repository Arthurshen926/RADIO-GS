import io
import torch
from radio_gs.querying.visual_scope_membership import (VisualMembershipHead,VisualMembershipRuntime,region_supervision,membership_loss,known_loss,balanced_query_risks,render_probabilities)
from radio_gs.querying.query_region_evidence import corrected_scores
from tests.test_query_region_evidence_next42 import fixture as old_fixture


def fixture():
    bank,_,_=old_fixture();head=VisualMembershipHead();return bank,head,VisualMembershipRuntime(bank,head)


def test_labels_deleted_permuted_exact():
    b,h,r=fixture();h.net[-1].weight.data.fill_(2);q=torch.eye(3);base=torch.zeros(4,3)
    expected=r.logits(q,base);clean={k:v for k,v in b.items() if k not in ['text','associations','records']}
    assert torch.equal(expected,VisualMembershipRuntime(clean,h).logits(q,base))
    b['text']=b['text'].flip(0);b['associations']=[[999],[],[7]]
    assert torch.equal(expected,VisualMembershipRuntime(b,h).logits(q,base))


def test_same_sign_classification_gradients():
    for initial in [3.,-3.]:
        logits=torch.full((2,1),initial,requires_grad=True)
        sup=dict(target=torch.tensor([[1.],[0.]]),valid=torch.ones(2,1,dtype=torch.bool),known_fraction=torch.ones(2,1))
        membership_loss(logits,sup).backward()
        assert logits.grad[0]<0 and logits.grad[1]>0
    _,h,r=fixture();h.net[-1].bias.data.fill_(3)
    logit=r.membership_logits(torch.eye(3)[:1]);logit.retain_grad()
    sup=dict(target=torch.tensor([[1.],[0.],[0.]]),valid=torch.ones(3,1,dtype=torch.bool),known_fraction=torch.ones(3,1))
    membership_loss(logit,sup).backward();assert (logit.grad[1:]>0).all() and h.net[-1].weight.grad.abs().sum()>0


def test_small_known_child_does_not_label_parent():
    a=torch.ones(100,2);a[20:,1]=0
    target=torch.zeros(100,1);target[:10]=1;known=target.bool()
    e=dict(target=target,known=known,visible=torch.ones(100,dtype=torch.bool))
    s=region_supervision(a,e,torch.ones(2,dtype=torch.bool))
    assert not s['valid'][0] and s['valid'][1] and s['target'][0]==1
    assert s['known_fraction'][0]==.1 and s['effective_pixels'][0]==100


def test_query_first_mixed_single_class_exact():
    pred=torch.tensor([[.8,.3],[.4,.9]])
    e=dict(target=torch.tensor([[1.,1.],[0.,1.]]),known=torch.tensor([[True,True],[True,False]]))
    # Query0 has P and N; query1 has only P. Average queries, not global classes.
    expected=((.2**2+.4**2)/2+.7**2)/2
    assert torch.allclose(known_loss(pred,e),torch.tensor(expected))
    e['known'].zero_();assert known_loss(pred,e) is None


def test_transport_qualified_negative_and_overlap():
    b,h,r=fixture();base=torch.zeros(4,1)
    delta,s=r.transport(torch.tensor([[3.],[-3.],[-3.]]),base)
    assert delta[0]>0 and delta[1]<0 and delta[2]==0 and delta[3]==0
    assert not r.transport(torch.full((3,1),-3.),base)[0].any()
    assert not r.transport(torch.tensor([[3.],[-3.],[-3.]]),base,0)[0].any()
    b['disjoint'].zero_();r=VisualMembershipRuntime(b,h)
    assert r.transport(torch.tensor([[3.],[-3.],[-3.]]),base)[0][1]==0


def test_zero_nontext_finite_serialization_batch():
    b,h,r=fixture();q=torch.eye(3);base=torch.tensor([[-100.,100.,0.]]*4)
    assert torch.equal(r.logits(q,base),base)
    assert torch.equal(corrected_scores(r,q,base.sigmoid(),base_logits=base),base.sigmoid())
    assert r.logits(q,base,'image') is base and r.logits(q,base,'prompt') is base
    h.net[-1].bias.data.fill_(3);result=r.logits(q,base);assert torch.isfinite(result).all()
    a=r.logits(q[:1],base[:,:1]);r.logits(q[1:2],base[:,1:2]);assert torch.equal(a,r.logits(q[:1],base[:,:1]))
    assert torch.allclose(result,torch.cat([r.logits(q[j:j+1],base[:,j:j+1]) for j in range(3)],1),atol=1e-6,rtol=0)
    buf=io.BytesIO();torch.save(h.state_dict(),buf);buf.seek(0);head=VisualMembershipHead();head.load_state_dict(torch.load(buf,weights_only=True));assert torch.equal(result,VisualMembershipRuntime(b,head).logits(q,base))
    b['weights'].zero_();assert torch.equal(VisualMembershipRuntime(b,h).logits(q,base),base)


def test_actual_render_order():
    op=torch.tensor([[.25,.75]]).to_sparse();logits=torch.tensor([[-4.],[2.]],requires_grad=True)
    pred=render_probabilities(op,logits)
    assert torch.equal(pred,torch.sparse.mm(op,logits.sigmoid()))
    assert not torch.allclose(pred,torch.sparse.mm(op,logits).sigmoid())
    pred.sum().backward();assert (logits.grad>0).all()
