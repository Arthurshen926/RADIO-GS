import pytest
import torch

from radio_gs.query_field import QueryPacket,Support
from radio_gs.query_field.field import CapabilityField
from radio_gs.query_field.storage import save_field,load_field
from test_query_preserving_field import example


def test_support_domain_changes_only_inferred_inputs_and_survives_export(tmp_path):
    field=example();semantic=field.semantic.codes[0]
    assert int(field.query(QueryPacket('text',semantic=semantic)).candidate_indices[0])==0
    packets=[QueryPacket('point_3d',positive=Support(torch.tensor([0]),torch.ones(1)),exact_positive=True),
             QueryPacket('point_2d',positive=Support(torch.tensor([0,1]),torch.tensor([.4,.6])),
                         negative=Support(torch.tensor([5]),torch.ones(1)))]
    expected=[field.query(packet).membership_scores.clone() for packet in packets]
    protected={key:value.clone() for key,value in field.state_dict().items()}
    field.anchor_eligible=torch.zeros(12,dtype=torch.bool);field.anchor_eligible[3:6]=True
    with torch.no_grad():field.affinity_bias.fill_(-2.)
    inferred=field.query(QueryPacket('image',semantic=semantic))
    assert inferred.candidate_indices.tolist()==[i for i in inferred.candidate_indices.tolist() if 3<=i<6]
    assert inferred.selected.all()  # The output domain is not cut to eligible anchors.
    with torch.no_grad():field.affinity_bias.copy_(protected['affinity_bias'])
    for packet,reference in zip(packets,expected):
        assert torch.equal(field.query(packet).membership_scores,reference)
    for key,value in protected.items():assert torch.equal(field.state_dict()[key],value)
    save_field(field,tmp_path/'scene',{});loaded,manifest=load_field(tmp_path/'scene')
    assert manifest['eligible_anchors']==3 and manifest['external_scene_dependencies']==[]
    for packet in packets+[QueryPacket('text',semantic=semantic,task='union')]:
        assert torch.equal(field.query(packet).membership_scores,loaded.query(packet).membership_scores)


def test_anchor_domain_rejects_unobserved_or_empty_support():
    original=example();geometry={key:getattr(original,key) for key in ['xyz','normals','quaternions','scales','opacities']}
    observed=original.observed.clone();observed[0]=False
    arguments=(geometry,original.semantic.codes,original.local.codes,original.group,observed)
    for eligible in [torch.zeros(12,dtype=torch.bool),torch.ones(12,dtype=torch.bool)]:
        with pytest.raises(ValueError,match='observed training support'):
            CapabilityField(*arguments,anchor_eligible=eligible)


def test_exposure_compiler_excludes_development_and_unknown_pixels(tmp_path):
    import json
    from radio_gs.query_field.storage import sha256
    from radio_gs.query_field.support_audit import compile_exposure
    masks=torch.zeros(2,9,12,dtype=torch.bool);masks[0,1:7,1:5]=True;masks[1,1:7,8:12]=True
    count=108;pixels=torch.arange(count)
    view=dict(masks=masks,scales_m=torch.ones(2),raw_mass=torch.ones(count),best_element=pixels,
              operator=torch.sparse_coo_tensor(torch.stack([pixels,pixels]),torch.ones(count),(count,count)).coalesce())
    torch.save(view,tmp_path/'view_00001.pt')
    manifest={'frames':[{'frame':1,'split':'train','compiled':{'sha256':sha256(tmp_path/'view_00001.pt')}},
                        {'frame':2,'split':'development','compiled':{'sha256':'not opened'}}]}
    (tmp_path/'observations.json').write_text(json.dumps(manifest))
    exposure=compile_exposure(tmp_path,torch.ones(count,dtype=torch.bool))
    assert torch.equal(exposure['mass'],torch.ones(count))
    assert exposure['eligible'].any() and not exposure['eligible'][0]
    assert not exposure['eligible'][~masks.any(0).flatten()].any()
