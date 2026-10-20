import pytest
import torch
from radio_gs.query_field.part_conditioning import integrate_views,condition_parts


def test_leave_one_view_out_never_self_protects_and_unknown_abstains():
    prior=torch.tensor([.7,.2]);p=torch.tensor([[.4],[0.]]);n=torch.tensor([[.6],[0.]])
    a,_=integrate_views(prior,p,n);b,protected=integrate_views(prior,p,n,protect=True)
    torch.testing.assert_close(a,b);torch.testing.assert_close(n,protected)
    assert b[1]==prior[1]


def test_independent_positive_protection_matches_literal_view_loop():
    p=torch.tensor([[.8,0.,.2],[0.,0.,0.],[.2,.3,0.]])
    n=torch.tensor([[0.,.9,.4],[0.,0.,.5],[.5,.2,.1]])
    prior=torch.tensor([.3,.7,.5]);expected=n.clone()
    for i in range(3):
        for v in range(3):
            ids=[j for j in range(3) if j!=v]
            denominator=(p[i,ids]+n[i,ids]).sum()
            expected[i,v]*=1-(p[i,ids].sum()/denominator if denominator>0 else 0)
    scores,actual=integrate_views(prior,p,n,protect=True)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(scores,(prior+p.sum(1))/(1+p.sum(1)+expected.sum(1)))
    control,_=integrate_views(prior,p,n)
    assert (scores>=control).all()  # Protection alone can also increase false members.


class Evidence:
    count=4;view_count=2;views=torch.tensor([0,0,1])
    def matrix(self,key):
        return {'positive':torch.tensor([[1.,0.,1.],[1.,0.,1.],[0.,1.,0.],[0.,0.,0.]]),
                'negative':torch.tensor([[0.,1.,0.],[0.,1.,0.],[1.,0.,1.],[0.,0.,0.]])}[key].to_sparse()


def test_conditioning_can_extend_a_supported_part_without_promoting_unknown():
    prior=torch.tensor([.9,.4,.1,.7]);available=torch.ones(4,dtype=torch.bool)
    score,d=condition_parts(Evidence(),prior,available,protect=True)
    assert score[1]>.5 and score[2]<.1 and score[3]==prior[3]
    assert d['admitted_regions']==2 and d['newly_selected']==1
    # A tiny positive margin must remain weak rather than becoming a full view.
    weak,_=condition_parts(Evidence(),torch.tensor([.50001,.50001,.1,.7]),available)
    assert (weak[:2]-.50001).abs().max()<1e-4
    with pytest.raises(ValueError):condition_parts(Evidence(),prior*2,available)


def test_native_function_precedes_spatial_mixing_and_shared_evidence_is_monotone():
    from radio_gs.query_field.part_conditioning import native_region_probability,prepare_parts,read_parts
    evidence=Evidence();evidence.region_semantic=torch.tensor([[1.,0.],[0.,1.],[0.,0.]])
    probability=native_region_probability(evidence,torch.tensor([1.,0.]),torch.tensor([[0.,1.]]))
    torch.testing.assert_close(probability,torch.tensor([10.,-10.,0.]).sigmoid())
    prior=torch.tensor([.2,.2,.9,.7]);available=torch.ones(4,dtype=torch.bool)
    prepared=prepare_parts(evidence,prior,available,region_probability=probability)
    a,da=read_parts(prior,available,prepared);b,db=read_parts(prior,available,prepared,protect=True)
    assert (b>=a).all() and a[1]>.5 and b[3]==prior[3]
    assert da['admitted_regions']==1 and da['positive_mass']==db['positive_mass']
    assert da['native_region_acceptance']
    with pytest.raises(ValueError):native_region_probability(evidence,torch.ones(3),torch.ones(1,3))


def test_native_state_export_is_self_contained_and_old_state_still_loads(tmp_path):
    from test_query_field_soft_paired_evidence import soft_fixture
    from test_query_preserving_field import example
    from radio_gs.query_field.soft_paired_evidence import enable
    from radio_gs.query_field.storage import save_field,load_field
    field=example();state=soft_fixture()
    for prefix in ['positive','negative','association']:
        pointer=state[prefix+'_pointer'];state[prefix+'_pointer']=torch.cat([pointer,pointer[-1].expand(8)])
    state['region_semantic']=torch.eye(3)
    enable(field,state);save_field(field,tmp_path/'field',{})
    loaded,manifest=load_field(tmp_path/'field')
    assert torch.equal(loaded.evidence.region_semantic,state['region_semantic'])
    assert manifest['external_scene_dependencies']==[]


def test_compiler_checks_source_indices_and_preserves_parent_state(tmp_path):
    import json
    from test_query_field_soft_paired_evidence import soft_fixture
    from test_query_preserving_field import example
    from radio_gs.query_field.soft_paired_evidence import enable
    from radio_gs.query_field.storage import save_field,load_field,sha256
    from radio_gs.query_field.compile_region_semantic import compile_field
    source=tmp_path/'source';source.mkdir();field=example();state=soft_fixture()
    for prefix in ['positive','negative','association']:
        p=state[prefix+'_pointer'];state[prefix+'_pointer']=torch.cat([p,p[-1].expand(8)])
    enable(field,state);frames=[];records=[];expected=[]
    for vi,frame in enumerate([3,7]):
        ids=[2,0] if vi==0 else [1]
        descriptors=torch.arange(3*field.semantic.native_dim,dtype=torch.float32).reshape(3,-1)+frame
        path=source/f'view_{frame:05d}.pt'
        torch.save(dict(frame=frame,semantic_descriptors=descriptors,scales_m=torch.ones(3)),path)
        frames.append(dict(frame=frame,split='train',compiled=dict(sha256=sha256(path))))
        records.append(dict(frame=frame,regions=ids));expected.append(descriptors[ids])
    (source/'observations.json').write_text(json.dumps(dict(frames=frames,world_units_per_metre=field.world_units_per_metre,
        splits=dict(train_frames=[3,7],development_frames=[9]))))
    parent=tmp_path/'parent';save_field(field,parent/'field',dict(observations_sha256=sha256(source/'observations.json')))
    construction=parent/'construction.json';construction.write_text(json.dumps(dict(source_records=records)))
    compile_field(parent/'field',source,construction,tmp_path/'export',legacy_v1=True)
    loaded,_=load_field(tmp_path/'export/field')
    assert torch.equal(loaded.evidence.region_semantic,torch.cat(expected))
    for key,value in field.state_dict().items():assert torch.equal(value,loaded.state_dict()[key])
    records[0]['frame']=9;construction.write_text(json.dumps(dict(source_records=records)))
    with pytest.raises(ValueError,match='illegal source frame'):
        compile_field(parent/'field',source,construction,tmp_path/'illegal',legacy_v1=True)


def test_native_half_storage_uses_float32_query_function():
    from radio_gs.query_field.part_conditioning import native_region_probability
    from torch.nn import functional as F
    evidence=Evidence();evidence.region_semantic=torch.tensor([[.136,.272,.447],[.631,.71,.814],[0.,0.,0.]],dtype=torch.float16)
    query=torch.tensor([.34412,.57123,.312]);negatives=torch.tensor([[.731,.912,.151]])
    d=F.normalize(evidence.region_semantic.float(),dim=1)
    expected=(10*(d@F.normalize(query,dim=0)-(d@F.normalize(negatives,dim=1).T).max(1).values)).sigmoid()
    actual=native_region_probability(evidence,query,negatives)
    assert actual.dtype==torch.float32 and torch.equal(actual,expected)
