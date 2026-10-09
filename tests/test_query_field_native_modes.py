import numpy as np
import pytest
from scipy import sparse
import torch
from torch.nn import functional as F

from radio_gs.query_field.native_modes import NativeModes
from radio_gs.query_field.contracts import QueryPacket
from radio_gs.query_field.runtime import QueryFieldRuntime
from radio_gs.query_field.storage import save_field,load_field
from radio_gs.querying.local_appearance_match import match_regions
from test_query_preserving_field import example


def state():
    rng=np.random.default_rng(192);atoms=rng.normal(size=(17,6)).astype(np.float32)
    w=sparse.csr_matrix(np.where(rng.random((7,17))>.7,rng.random((7,17)),0).astype(np.float32))
    w=(sparse.diags(1/np.asarray(w.sum(1)).ravel())@w).tocsr();w.sort_indices()
    normalized=F.normalize(torch.from_numpy(atoms),dim=1)
    return dict(anchors=torch.arange(7),semantic=torch.randn(7,9),atoms=normalized,
        pointer=torch.tensor(w.indptr,dtype=torch.long),index=torch.tensor(w.indices,dtype=torch.long),
        mass=torch.from_numpy(w.data),mean=torch.from_numpy(w@normalized.numpy()),relation_columns=torch.arange(7)),atoms,w


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_weighted_function_matches_actual_reference_and_chunk_tail(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    s,atoms,w=state();q=np.random.default_rng(32).normal(size=(5,6)).astype(np.float32);mass=np.array([1,0,2,4,1],np.float32)
    expected=match_regions(atoms,w,np.array([0,17]),np.array([0,7]),[q],[mass],device=device)['chamfer'][:,0]
    m=NativeModes(s,12).to(device)
    for chunk in [1,3,16]:
        actual=m.local_scores(torch.from_numpy(q),torch.from_numpy(mass),chunk=chunk,query_chunk=2).cpu().numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=0)


def test_equal_means_do_not_preserve_local_function():
    v=3**.5/2;atoms=torch.tensor([[v,0,.5],[-v,0,.5],[0,v,.5],[0,-v,.5]])
    s=dict(anchors=torch.arange(2),semantic=torch.ones(2,9),atoms=atoms,
        pointer=torch.tensor([0,2,4]),index=torch.arange(4),mass=torch.full((4,),.5),
        mean=torch.tensor([[0,0,.5],[0,0,.5]]),relation_columns=torch.arange(2))
    m=NativeModes(s,12);q=torch.tensor([[1.,0,0]])
    torch.testing.assert_close(m.local_scores(q),torch.tensor([v/2,0.]))
    assert torch.equal(m.local_scores(q,rule='mean'),torch.zeros(2))


def test_export_all_modes_query_and_registered_parent_unchanged(tmp_path):
    f=example();s,_,_=state();f.native_modes=NativeModes(s,12);f.native_mode_rule='measure'
    packet=QueryPacket('image',semantic=torch.randn(9),local=torch.randn(5,6),local_weights=torch.tensor([1.,0,2,3,1]))
    old=QueryFieldRuntime(f).query(packet);save_field(f,tmp_path/'field',{})
    g,_=load_field(tmp_path/'field');new=QueryFieldRuntime(g).query(packet)
    assert torch.equal(old.membership_scores,new.membership_scores)
    assert len(new.candidate_indices)==7 and f.candidate_count==3
    from radio_gs.query_field.adapters import point_3d
    assert torch.equal(QueryFieldRuntime(f).query(point_3d(f,[2,0,0])).membership_scores,
                       example().query(point_3d(example(),[2,0,0])).membership_scores)


def test_query_padding_mass_is_required_and_validated():
    q=QueryPacket('image',semantic=torch.ones(9),local=torch.ones(2,6),local_weights=torch.zeros(2))
    with pytest.raises(ValueError,match='weights'):q.validate(12,9,6)


def test_native_input_keeps_resolution_and_fractional_padding_weights():
    from PIL import Image
    from radio_gs.query_field.adapters import NativeQueryEncoder
    class Vision:
        def encode(self,x):return torch.ones(1,9)
    class Local:
        def patches(self,image):
            self.size=image.size
            return torch.ones(2,6),torch.tensor([1.,.25])
    encoder=NativeQueryEncoder('unused','cpu');encoder._vision=Vision();encoder._dino=Local()
    p=encoder.image(Image.new('RGB',(21,15)),native_resolution=True)
    assert encoder._dino.size==(21,15) and torch.equal(p.local_weights,torch.tensor([1.,.25]))
    p=encoder.image(Image.new('RGB',(21,15)))
    assert encoder._dino.size==(448,320) and p.local_weights is None
