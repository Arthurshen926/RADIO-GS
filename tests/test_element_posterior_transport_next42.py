import io
import torch
from radio_gs.querying.element_posterior_transport import ElementPosteriorRuntime,camera_groups
from radio_gs.querying.visual_scope_membership import VisualMembershipRuntime,VisualMembershipHead
from radio_gs.querying.query_region_evidence import corrected_scores


def fixture(n=10,views=(0,),groups=None):
    k=len(views);ii=torch.arange(n).repeat(k);kk=torch.arange(k).repeat_interleave(n)
    bank=dict(indices=torch.stack([ii,kk]),weights=torch.ones(n*k),shape=[n,k],views=torch.tensor(views),held_views=[],quality=torch.ones(k),area=torch.ones(k),disjoint=torch.zeros(k,k,dtype=torch.bool),text=torch.randn(k,3),associations=[[1]]*k)
    bank.update({name:torch.eye(3)[torch.arange(k)%3] for name in ['C','F','B','tight']})
    meta=dict(keep=torch.ones(k,dtype=torch.bool),footprint_weights=torch.ones(n*k),region_groups=torch.tensor(groups if groups is not None else views))
    head=VisualMembershipHead();return bank,head,meta,ElementPosteriorRuntime(bank,head,meta)


def test_low_member_hole_and_convex_target():
    b,h,m,r=fixture();parent=torch.logit(torch.tensor([.999]*9+[.01]))[:,None];member=torch.logit(torch.tensor([[.9]]))
    old=VisualMembershipRuntime(b,h).transport(member,parent)[0];assert not old.any()
    new,s=r.transport(member,parent);assert new[-1]>parent[-1] and torch.equal(new[:-1],parent[:-1])
    assert torch.allclose(new[-1].sigmoid(),torch.tensor([(.01+.8*.9)/1.8]))
    assert ((s['pooled_probability']>=0)&(s['pooled_probability']<=1)).all()


def test_missing_unknown_views_do_not_dilute():
    b,h,m,r=fixture(2);parent=torch.zeros(2,1);expected=r.transport(torch.tensor([[2.]]),parent)[0]
    b,h,m,r=fixture(2,(0,1),(0,1));actual=r.transport(torch.tensor([[2.],[0.]]),parent)[0]
    assert torch.equal(expected,actual)


def test_same_view_duplicate_and_same_pose_mass_bounded():
    _,_,_,r=fixture(2);base=torch.zeros(2,1);expected=r.transport(torch.tensor([[2.]]),base)[0]
    b,h,m,r=fixture(2,(0,0),(0,0));m['keep'][1]=False;r=ElementPosteriorRuntime(b,h,m)
    assert torch.equal(expected,r.transport(torch.tensor([[2.],[2.]]),base)[0])
    b,h,m,r=fixture(2,tuple(range(24)),[0]*24);new,s=r.transport(torch.full((24,1),2.),base)
    assert torch.equal(expected,new) and s['group_mass'].max()<=1
    poses=torch.eye(4).repeat(24,1,1);g,_=camera_groups(poses);assert len(g.unique())==1


def test_negative_requires_same_view_disjoint_positive():
    b,h,m,r=fixture(2,(0,0),(0,0));base=torch.zeros(2,1)
    assert torch.equal(r.transport(torch.full((2,1),-2.),base)[0],base)
    # Separate region supports, disjoint within view.
    b['indices']=torch.tensor([[0,1],[0,1]]);b['weights']=torch.ones(2);m['footprint_weights']=torch.ones(2);b['disjoint']=torch.tensor([[False,True],[True,False]])
    r=ElementPosteriorRuntime(b,h,m);new,_=r.transport(torch.tensor([[2.],[-2.]]),base)
    assert new[0]>0 and new[1]<0
    b['views'][1]=1;r=ElementPosteriorRuntime(b,h,m);assert r.transport(torch.tensor([[2.],[-2.]]),base)[0][1]==0


def test_neutral_zero_extreme_nontext_exact():
    b,h,m,r=fixture(4);base=torch.tensor([[-150.],[-100.],[100.],[150.]])
    assert torch.equal(r.transport(torch.zeros(1,1),base)[0],base)
    q=torch.eye(3)[:1];assert r.logits(q,base,'image') is base and r.logits(q,base,'prompt') is base
    assert torch.equal(corrected_scores(r,q,base.sigmoid(),base_logits=base),base.sigmoid())
    h.net[-1].bias.data.fill_(2);m['footprint_weights'].zero_();r=ElementPosteriorRuntime(b,h,m);assert torch.equal(r.logits(q,base),base)


def test_labels_serialization_batch_aba_and_output_probability():
    b,h,m,r=fixture(4);h.net[-1].bias.data.fill_(2);base=torch.tensor([[-150.,2.,0.]]*4);q=torch.eye(3)
    expected=r.logits(q,base);assert torch.isfinite(expected).all()
    clean={k:v for k,v in b.items() if k not in ['text','associations']};assert torch.equal(expected,ElementPosteriorRuntime(clean,h,m).logits(q,base))
    b['text'].normal_();b['associations']=[[]];assert torch.equal(expected,ElementPosteriorRuntime(b,h,m).logits(q,base))
    a=r.logits(q[:1],base[:,:1]);r.logits(q[1:2],base[:,1:2]);assert torch.equal(a,r.logits(q[:1],base[:,:1]))
    assert torch.allclose(expected,torch.cat([r.logits(q[j:j+1],base[:,j:j+1]) for j in range(3)],1),atol=2e-6,rtol=0)
    buf=io.BytesIO();torch.save(dict(bank=b,meta=m,head=h.state_dict()),buf);buf.seek(0);saved=torch.load(buf,weights_only=False);fresh=VisualMembershipHead();fresh.load_state_dict(saved['head']);assert torch.equal(expected,ElementPosteriorRuntime(saved['bank'],fresh,saved['meta']).logits(q,base))
    pooled=r.transport(r.membership_logits(q),base)[1]['pooled_probability'];assert torch.allclose(expected.sigmoid().double(),pooled,atol=1e-7,rtol=0)
    assert torch.allclose(corrected_scores(r,q,base.sigmoid(),base_logits=base),pooled.float(),atol=1e-7,rtol=0)


def test_six_group_convex_bounds_and_camera_determinism():
    poses=torch.eye(4).repeat(12,1,1);poses[:,0,3]=torch.arange(12);a,_=camera_groups(poses);b,_=camera_groups(poses);assert torch.equal(a,b) and len(a.unique())==6
    _,_,_,r=fixture(4,tuple(range(6)),list(range(6)));base=torch.tensor([[-150.],[150.],[-2.],[2.]])
    logits,s=r.transport(torch.linspace(.1,4,6)[:,None],base);p=s['pooled_probability'];assert ((p>=0)&(p<=1)).all() and s['group_mass'].sum(0).max()<=6 and torch.isfinite(logits).all()
