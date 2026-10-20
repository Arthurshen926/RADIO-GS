import copy
import json

import numpy as np
import pytest
import torch

from radio_gs.evidence_core.gaussian_observations import compile_gaussian_view
from radio_gs.evidence_core.provenance import (attach_provenance, field_binding,
    file_sha256, verify_native_block, verify_replacement_lineage)
from radio_gs.evidence_core.store import CompiledRegionEvidence


def field():
    return dict(descriptors=torch.eye(2,dtype=torch.float16),
        crow=torch.tensor([0,2,2,3],dtype=torch.int32),columns=torch.tensor([0,1,1],dtype=torch.int32),
        weights=torch.tensor([.25,.75,1.]),supported=torch.tensor([True,False,True]),
        shape=[3,2],geometry=dict(path='geometry.pt',sha256='a'*64))


def sidecar(tmp_path,payload):
    field_path=tmp_path/'field.pt';torch.save(payload,field_path)
    record=dict(schema='radiogs.compiled_region_provenance.v1',
        field=dict(path=str(field_path),binding=field_binding(payload)),
        source_frames=[12,18,24],region_offsets=[0,1,1,2],
        regions_by_view=[dict(frame=12,proposal_indices=[5]),dict(frame=18,proposal_indices=[]),
                         dict(frame=24,proposal_indices=[3])],verification='fixture')
    path=tmp_path/'provenance.json';path.write_text(json.dumps(record))
    return field_path,dict(path=str(path),sha256=file_sha256(path)),record


def test_overlay_preserves_tensors_empty_views_and_never_invents_visibility(tmp_path):
    payload=field();path,reference,_=sidecar(tmp_path,payload)
    result=attach_provenance(payload,path,reference,source_frames=[12,18,24])
    for key in ('descriptors','crow','columns','weights','supported'):
        assert result[key] is payload[key]
    assert 'region_offsets' not in payload
    store=CompiledRegionEvidence.from_payload(result)
    np.testing.assert_array_equal(store.region_frames,[0,2])
    assert store.visibility('cpu') is None
    assert not store.independent_visibility_available


@pytest.mark.parametrize('key',['descriptors','columns','weights','supported'])
def test_overlay_rejects_changed_same_shape_field(tmp_path,key):
    payload=field();path,reference,_=sidecar(tmp_path,payload)
    changed=copy.deepcopy(payload)
    if key=='supported':changed[key][0]=False
    else:changed[key].view(-1)[0]+=1
    with pytest.raises(ValueError,match='field content'):
        attach_provenance(changed,path,reference)


def test_overlay_rejects_changed_metadata_source_order_and_conflicting_inline_provenance(tmp_path):
    payload=field();path,reference,record=sidecar(tmp_path,payload)
    with pytest.raises(ValueError,match='source view order'):
        attach_provenance(payload,path,reference,source_frames=[24,18,12])
    with pytest.raises(ValueError,match='inline and sidecar'):
        attach_provenance(dict(payload,source_frames=[12,18,24],region_offsets=torch.tensor([0,0,1,2])),path,reference)
    record['regions_by_view'][0]['proposal_indices']=[7]
    from pathlib import Path
    Path(reference['path']).write_text(json.dumps(record))
    with pytest.raises(ValueError,match='file hash'):
        attach_provenance(payload,path,reference)


def test_same_count_wrong_region_order_is_rejected():
    payload=field()
    row=dict(frame=12,descriptors=payload['descriptors'].flip(0),native_proposal_indices=torch.tensor([5,3]))
    with pytest.raises(ValueError,match='descriptor order/content'):
        verify_native_block(payload,row,12,0)
    row['descriptors']=payload['descriptors']
    assert verify_native_block(payload,row,12,0)==[5,3]
    with pytest.raises(ValueError,match='frame identity'):verify_native_block(payload,row,24,0)


def test_replacement_lineage_requires_exact_operator_and_recorded_view_order():
    parent=field();replacement=copy.deepcopy(parent)
    replacement['descriptors']=torch.ones(2,7,dtype=torch.float16)
    records=[dict(frame=12,regions=1),dict(frame=24,regions=1)]
    verify_replacement_lineage(parent,replacement,records,[12,24],[0,1,2])
    with pytest.raises(ValueError,match='order/count'):
        verify_replacement_lineage(parent,replacement,records[::-1],[12,24],[0,1,2])
    replacement['columns']=replacement['columns'].flip(0)
    with pytest.raises(ValueError,match='transfer differs'):
        verify_replacement_lineage(parent,replacement,records,[12,24],[0,1,2])


def hit():
    # Element 1 is visible only outside regions; element 3 is fully occluded.
    # Duplicate pixel/element hits exercise the original coalescing order.
    return dict(pixel_ids=torch.tensor([0,0,1,2,3]),gaussian_ids=torch.tensor([0,0,2,1,2]),
                weights=torch.tensor([.25,.125,.5,.75,.125]))


def old_compile(h,m,n):
    count=m.sum(0);p,e=h['pixel_ids'].long(),h['gaussian_ids'].long()
    a=torch.sparse_coo_tensor(torch.stack([e,p]),h['weights'].float()*(count[p]>0),(n,m.shape[1])).coalesce()
    return torch.sparse.mm(a,(m/count.clamp_min(1)).T),torch.sparse.mm(a,torch.ones(m.shape[1],1))


@pytest.mark.parametrize('regions',[0,1,3])
def test_gaussian_compiler_keeps_overlap_mass_and_independent_visibility(regions):
    m=torch.tensor([[1.,0.,0.,0.],[1.,1.,0.,0.],[0.,1.,0.,0.]])[:regions]
    old,mass=old_compile(hit(),m,4)
    new=compile_gaussian_view(hit(),m,4,retain_visibility=True)
    assert torch.equal(old,new.region_contributions)
    assert torch.equal(mass,new.supported_mass)
    assert torch.equal(new.geometric_mass,torch.tensor([.375,.75,.625,0.]))
    assert new.supported_mass[1]==0 and new.geometric_mass[1]>0
    if regions==3:
        torch.testing.assert_close(new.region_contributions[0],torch.tensor([.1875,.1875,0.]))


def test_empty_hits_and_no_visibility_request():
    empty={k:v[:0] for k,v in hit().items()}
    result=compile_gaussian_view(empty,torch.zeros(2,4),3)
    assert result.geometric_mass is None and not result.supported_mass.any()
    assert result.region_contributions.shape==(3,2)


def test_geometric_visibility_is_a_separate_validated_relation():
    payload=field();payload.update(source_frames=[12,24],region_offsets=torch.tensor([0,1,2]))
    visibility=dict(crow=torch.tensor([0,1,2,3]),columns=torch.tensor([0,0,1]),
        weights=torch.tensor([.3,.7,.5]),shape=[3,2],raster_hw=[[3,4],[3,4]],kind='unconditioned_compositing_mass')
    payload['geometric_visibility']=visibility
    store=CompiledRegionEvidence.from_payload(payload)
    assert store.independent_visibility_available
    assert store.visibility('cpu').to_dense()[1,0]==.7 and not payload['supported'][1]
    visibility['shape']=[3,3]
    with pytest.raises(ValueError,match='visibility contract'):CompiledRegionEvidence.from_payload(payload)


def test_real_readout_consumes_overlay_without_reopening_source_assets(tmp_path,monkeypatch):
    from radio_gs.querying.native_observation_prior import compiled_observation_prior
    payload=field();path,reference,_=sidecar(tmp_path,payload)
    monkeypatch.setenv('RADIO_GS_EVIDENCE_CORE','shared_v1')
    scene=dict(native_observation_field=str(path),source_frames=[12,18,24],
               native_observation_reliability='heldout_view_support')
    query=torch.eye(2);fallback=torch.zeros(3,2)
    inline=dict(payload,source_frames=[12,18,24],region_offsets=torch.tensor([0,1,1,2]))
    expected=compiled_observation_prior(scene,query,payload['supported'],fallback,field=inline)[0]
    # There are no source caches, RGB or cameras in this fixture. Deserialization
    # of a large map is also prohibited when the caller already supplies it.
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('unexpected source or field read'))
    result,report=compiled_observation_prior(dict(scene,native_observation_provenance=reference),
        query,payload['supported'],fallback,field=payload)
    assert torch.equal(result,expected)
    assert report['region_provenance']['sha256']==reference['sha256']
    assert report['source_rgb_read'] is False
