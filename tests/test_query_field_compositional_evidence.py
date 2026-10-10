import numpy as np
import pytest
import torch
from scipy import sparse
from radio_gs.query_field.compositional_evidence import compile_view,CompositionalEvidence


def test_union_before_lifting_retains_complementary_subpixel_support():
    transfer=sparse.csr_matrix(np.array([[.5,.5],[.2,.8]],np.float32))
    p=np.eye(2,dtype=bool);n=~p
    view=compile_view(transfer,p,n,[0,1]);model=CompositionalEvidence(dict(count=2,region_count=2,views=[view]))
    pos,neg,_=model.prepare(torch.tensor([True,True]))
    torch.testing.assert_close(pos,torch.ones(2,1));torch.testing.assert_close(neg,torch.zeros(2,1))
    assert (transfer@p.T).max(1).max()<1 # Marginal max cannot recover union mass.
    for index in range(2):
        selected=torch.arange(2)==index;a,b,_=model.prepare(selected)
        torch.testing.assert_close(a[:,0],torch.from_numpy((transfer@p[index].astype(np.float32))))
        torch.testing.assert_close(b[:,0],torch.from_numpy((transfer@n[index].astype(np.float32))))


def test_unknown_empty_and_duplicate_regions_do_not_create_negative_evidence():
    t=sparse.eye(3,dtype=np.float32,format='csr')
    p=np.array([[1,0,0],[1,0,0]],bool);n=np.array([[0,1,1],[0,1,0]],bool)
    model=CompositionalEvidence(dict(count=3,region_count=2,views=[compile_view(t,p,n,[0,1])]))
    a,b,d=model.prepare(torch.tensor([True,True]))
    torch.testing.assert_close(a[:,0],torch.tensor([1.,0.,0.]));torch.testing.assert_close(b[:,0],torch.tensor([0.,1.,0.]))
    a,b,d=model.prepare(torch.zeros(2,dtype=torch.bool));assert not a.any() and not b.any() and d['active_views']==0
    with pytest.raises(ValueError):model.prepare(torch.ones(2))


def test_atom_preparation_integrates_with_shared_member_decoder():
    from radio_gs.query_field.part_conditioning import read_parts
    t=sparse.eye(2,dtype=np.float32,format='csr');p=np.array([[1,0]],bool)
    model=CompositionalEvidence(dict(count=2,region_count=1,views=[compile_view(t,p,~p,[0])]))
    prior=torch.tensor([.3,.7]);available=torch.ones(2,dtype=torch.bool)
    result,details=read_parts(prior,available,model.prepare(torch.tensor([True])),protect=True)
    torch.testing.assert_close(result,torch.tensor([.65,.35]));assert not details['source_access']


def test_all_acceptance_subsets_match_direct_partial_union_across_views():
    from itertools import product
    rng=np.random.default_rng(19);states=[];direct=[]
    for vi in range(2):
        relation=rng.integers(0,3,(5,9));relation[:,-1]=relation[:,0]
        p=relation==1;n=relation==2
        t=rng.random((3,9),dtype=np.float32);t/=t.sum(1,keepdims=True)
        states.append(compile_view(sparse.csr_matrix(t),p,n,range(vi*5,(vi+1)*5)))
        direct.append((t,p,n))
    model=CompositionalEvidence(dict(count=3,region_count=10,views=states))
    for bits in product([False,True],repeat=5):
        accepted=torch.tensor(list(bits)+list(reversed(bits)))
        a,b,_=model.prepare(accepted)
        for vi,(t,p,n) in enumerate(direct):
            chosen=accepted[vi*5:(vi+1)*5].numpy()
            ep=t@p[chosen].any(0).astype(np.float32) if chosen.any() else np.zeros(3,np.float32)
            en=t@n[chosen].all(0).astype(np.float32) if chosen.any() else np.zeros(3,np.float32)
            torch.testing.assert_close(a[:,vi],torch.from_numpy(ep));torch.testing.assert_close(b[:,vi],torch.from_numpy(en))
