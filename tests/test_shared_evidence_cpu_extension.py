import hashlib

import numpy as np
import pytest
from scipy import sparse
import torch

from radio_gs import evidence_core as core
from radio_gs.evidence_core.query_contracts import authorized_crop_path, validate_public_query
from radio_gs.evidence_core.store import CompiledRegionEvidence
from radio_gs.querying.direct_anchor_membership import DirectAnchorMembership


@pytest.mark.parametrize('backend',['numpy','torch'])
def test_misaligned_permission_cannot_expand_pixel_axis(backend):
    p=np.array([True,False,True]);v=np.ones((3,1),bool)
    if backend=='torch':p=torch.from_numpy(p);v=torch.from_numpy(v)
    with pytest.raises(ValueError,match='shape'):
        core.compile_domains(p,~p,visible=v,negative_origin='model_complement')
    d=core.compile_domains(p,~p,visible=v[:,0],negative_origin='model_complement')
    with pytest.raises(ValueError,match='shape'):
        core.signed_weights(d,v,1.)


def test_view_balance_preserves_overlap_and_unsupported_elements():
    m=sparse.csr_matrix(np.array([[1,1,1],[1,0,0],[0,0,0]],np.float32))
    t=core.view_balanced_transfer(m,np.array([0,2,3]))
    np.testing.assert_array_equal(t.toarray(),[[.25,.25,.5],[1.,0.,0.],[0.,0.,0.]])
    # Splitting one view into independent views would change .25/.25/.5 to 1/3.
    assert not np.allclose(t.toarray()[0],[1/3]*3)


def field():
    return dict(descriptors=torch.eye(2),crow=torch.tensor([0,2,2],dtype=torch.int32),
        columns=torch.tensor([0,1],dtype=torch.int32),weights=torch.tensor([.75,.25]),
        supported=torch.tensor([True,False]),shape=[2,2])


def test_compiled_adapter_does_not_invent_missing_visibility_or_source_views():
    f=field();r=CompiledRegionEvidence.from_payload(f)
    assert r.payload is f and r.region_frames is None and not r.independent_visibility_available
    f['source_frames']=[4,8]
    with pytest.raises(ValueError,match='partial region provenance'):
        CompiledRegionEvidence.from_payload(f)
    f['region_offsets']=torch.tensor([0,1,2])
    assert CompiledRegionEvidence.from_payload(f).region_frames.tolist()==[0,1]


def test_category_competes_before_transport_and_consumes_shared_operator(tmp_path,monkeypatch):
    from radio_gs.querying.native_observation_prior import compiled_observation_prior
    monkeypatch.setenv('RADIO_GS_EVIDENCE_CORE','shared_v1')
    path=tmp_path/'field.pt';path.write_bytes(b'referenced asset')
    scene=dict(native_observation_field=str(path));f=field()
    fallback=torch.tensor([[.3,-.3],[.7,-.7]])
    def solve():return compiled_observation_prior(scene,torch.eye(2),f['supported'],fallback,temperature=8.,field=f)[0]
    value=solve()
    expected=torch.tensor([[.75,.25]])@torch.softmax(8*torch.eye(2),dim=-1)
    torch.testing.assert_close(value[0].sigmoid(),expected[0])
    assert torch.equal(value[1],fallback[1])
    assert not torch.allclose(value[0].sigmoid(),torch.softmax(torch.tensor([6.,2.]),dim=0))
    monkeypatch.setattr(core,'project',lambda operator,x:torch.zeros(operator.shape[0],x.shape[1]))
    assert not torch.equal(value,solve())


@pytest.mark.parametrize('solver',['consensus','rendered','selection_transport'])
def test_response_paths_consume_shared_projection(monkeypatch,solver):
    from radio_gs.querying.response_refinement import fuse_admitted_evidence
    from radio_gs.querying.response_reconstruction import solve_rendered_evidence
    from radio_gs.pipeline.subset import _transport
    monkeypatch.setenv('RADIO_GS_EVIDENCE_CORE','shared_v1')
    op=torch.eye(2).to_sparse();initial=torch.full((2,1),.5);y=torch.tensor([[1.],[0.]]);k=torch.ones_like(y)
    def solve():
        if solver=='consensus':return fuse_admitted_evidence(initial,[(op,y,k)])[0]
        if solver=='rendered':return solve_rendered_evidence(initial,[(op,y,k,1.)])[0]
        return _transport(op,y)
    old=solve()
    monkeypatch.setattr(core,'project',lambda operator,x:torch.zeros(operator.shape[0],x.shape[1]))
    assert not torch.equal(old,solve())


def member_reader(monkeypatch,chunk_size=None):
    class Head:
        sizes=[]
        def predict_proba(self,x):
            self.sizes.append(len(x));p=np.clip(x[:,0],0,1)
            return np.column_stack([1-p,p])
    head=Head();monkeypatch.setattr('joblib.load',lambda _:dict(estimator=head,threshold=.35,condition='test'))
    m=sparse.csr_matrix(np.array([[1,1,0],[1,0,0],[0,1,0],[1,1,0],[0,0,0]],np.float32))
    v=sparse.csr_matrix(np.array([[1],[1],[1],[1],[0]],np.float32))
    transfer=core.view_balanced_transfer(m,np.array([0,3]))
    reader=DirectAnchorMembership(m,v,np.zeros(3,dtype=int),np.arange(15,dtype=np.float32).reshape(5,3),
        np.eye(3,dtype=np.float32),transfer,'fixture',chunk_size=chunk_size)
    return reader,head


def test_chunked_head_visits_all_elements_and_matches_full_output(monkeypatch):
    monkeypatch.setenv('RADIO_GS_EVIDENCE_CORE','shared_v1')
    full,_=member_reader(monkeypatch);chunked,head=member_reader(monkeypatch,2)
    old=full.infer(0);new=chunked.infer(0)
    for i in [0,1]:np.testing.assert_array_equal(old[i],new[i])
    assert max(head.sizes)==2 and sum(head.sizes)==5
    assert new[0][0]==1 and len(new[1])==5
    empty=chunked.infer_region(2)
    assert len(empty[1])==5 and not empty[1].any() and empty[2]['anchor_vertex'] is None


def test_member_features_consume_shared_grouping(monkeypatch):
    monkeypatch.setenv('RADIO_GS_EVIDENCE_CORE','shared_v1')
    reader,_=member_reader(monkeypatch);old=reader.infer(0)[0]
    monkeypatch.setattr(core,'grouped_membership_features',lambda m,v,f,a,ids:np.zeros((len(ids),6),np.float32))
    assert not np.array_equal(old,reader.infer(0)[0])


@pytest.mark.parametrize('size',[0,-2,True,2.5])
def test_invalid_chunk_size_is_rejected_before_loading_checkpoint(monkeypatch,size):
    with pytest.raises(ValueError,match='chunk size'):member_reader(monkeypatch,size)


def test_public_queries_reject_extra_image_pose_and_point_rgb():
    image=dict(query_id='q',scene_id='s',modality='image',crop_rgb_path='assets/crop.png',crop_rgb_sha256='a'*64)
    validate_public_query(image)
    with pytest.raises(ValueError,match='undeclared'):
        validate_public_query(dict(image,camera_to_world=np.eye(4).tolist()))
    point=dict(query_id='q',scene_id='s',modality='point_3d',point_world_xyz=[0,0,0])
    assert not validate_public_query(point)['query_rgb_crop']
    with pytest.raises(ValueError,match='undeclared'):validate_public_query(dict(point,source_rgb='frame.jpg'))


def test_crop_access_binds_public_asset_contents(tmp_path):
    path=tmp_path/'crop.png';path.write_bytes(b'public query crop')
    query=dict(query_id='q',scene_id='s',modality='image',crop_rgb_path=path.name,
               crop_rgb_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert authorized_crop_path(tmp_path,query)==path
    path.write_bytes(b'replaced')
    with pytest.raises(ValueError,match='hash'):authorized_crop_path(tmp_path,query)
    with pytest.raises(ValueError,match='within'):authorized_crop_path(tmp_path,dict(query,crop_rgb_path='../frame.png'))
