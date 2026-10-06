import json
import shutil

import pytest
import torch
from torch.nn import functional as F

from radio_gs.query_field import CapabilityField, QueryPacket, Support
from radio_gs.query_field.field import NativeCoordinates
from radio_gs.query_field.adapters import pixel_support, point_3d
from radio_gs.query_field.observations import partial_mask_target
from radio_gs.query_field.storage import save_field, load_field
from radio_gs.query_field.train import partial_membership_loss


def example():
    torch.manual_seed(17)
    n=12
    xyz=torch.stack([torch.arange(n),torch.zeros(n),torch.zeros(n)],dim=1).float()
    geometry=dict(xyz=xyz,normals=torch.tensor([[0.,0.,1.]]).repeat(n,1),
        quaternions=torch.tensor([[1.,0.,0.,0.]]).repeat(n,1),
        scales=torch.full((n,3),.02),opacities=torch.full((n,1),.8))
    return CapabilityField(geometry,torch.randn(n,9),torch.randn(n,6),torch.randn(n,4),
                           torch.ones(n,dtype=torch.bool),candidate_count=3)


def test_affine_queries_match_explicit_native_space_with_bias_and_chunking():
    torch.manual_seed(5)
    codes=torch.randn(27,4);weight=torch.randn(9,4);bias=torch.randn(9);query=torch.randn(5,9)
    codec=NativeCoordinates(codes,weight,bias)
    expected=F.normalize(F.linear(codes,weight,bias),dim=-1)@F.normalize(query,dim=-1).T
    torch.testing.assert_close(codec.score(query,chunk_size=3),expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(codec.score(query,chunk_size=100),expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(codec.score(query,chunk_size=3,patch_max=True),expected.max(1).values,atol=1e-6,rtol=1e-5)
    # The old transpose-only cosine is a genuinely different operation.
    wrong=F.normalize(codes,dim=-1)@F.normalize(query@weight,dim=-1).T
    assert (expected-wrong).abs().max()>.1


def test_training_cannot_rewrite_native_identity_or_geometry():
    field=example()
    before={k:v.clone() for k,v in field.named_buffers()}
    original=field.group.detach().clone()
    optimizer=torch.optim.Adam(field.parameters(),lr=.1)
    logits=field.member_logits(Support(torch.tensor([0]),torch.ones(1)),.2)
    loss=F.binary_cross_entropy_with_logits(logits[1:],torch.ones(11))
    loss.backward();optimizer.step()
    assert not torch.equal(original,field.group)
    for key,value in field.named_buffers():assert torch.equal(before[key],value),key


def test_unknown_and_nested_masks_never_supply_exclusion_gradients():
    masks=torch.zeros(3,9,12,dtype=torch.bool)
    masks[0,1:7,1:5]=True
    masks[1,0:8,0:6]=True  # enclosing parent
    masks[2,1:7,8:12]=True # comparable disjoint region
    positive,negative=partial_mask_target(masks,0,torch.tensor([1.,2.,1.]))
    assert not negative[masks[1]].any()
    assert negative[masks[2]].any()
    values=torch.full((9,12),.4,requires_grad=True)
    partial_membership_loss(values,positive,negative).backward()
    assert torch.equal(values.grad[~(positive|negative)],torch.zeros_like(values.grad[~(positive|negative)]))
    assert (values.grad[positive]<0).all() and (values.grad[negative]>0).all()


def test_pixel_prompt_keeps_all_contributors_and_has_no_rgb_requirement():
    operator=torch.sparse_coo_tensor(torch.tensor([[0,0,1],[1,3,2]]),torch.tensor([.2,.8,1.]),(2,12)).coalesce()
    support=pixel_support(operator,0)
    assert support.indices.tolist()==[1,3]
    torch.testing.assert_close(support.weights,torch.tensor([.2,.8]))
    result=example().query(QueryPacket('point_2d',positive=support))
    assert not result.diagnostics['hard_positive_constraint']


def test_estimated_identity_is_not_clamped_positive_and_modes_share_core():
    field=example()
    with torch.no_grad():field.affinity_bias.fill_(1.2)
    for mode in ['text','image']:
        result=field.query(QueryPacket(mode,semantic=field.semantic.codes[0]))
        assert result.membership_scores.max()<.5
        assert not result.selected.any()
        assert result.diagnostics['shared_member_core']
        assert len(result.candidate_weights)==3
    packet=point_3d(field,field.xyz[0])
    assert field.query(packet).selected[0]
    with pytest.raises(ValueError,match='not exact'):
        field.query(QueryPacket('image',semantic=field.semantic.codes[0],
             positive=Support(torch.tensor([0]),torch.ones(1)),exact_positive=True))


def test_cold_export_is_self_contained_and_rejects_tampered_payload(tmp_path):
    field=example();packet=QueryPacket('image',semantic=field.semantic.codes[2],local=field.local.codes[2:4])
    expected=field.query(packet)
    save_field(field,tmp_path/'source',dict(training_root='/unavailable/source/cache'))
    shutil.copytree(tmp_path/'source',tmp_path/'isolated');shutil.rmtree(tmp_path/'source')
    loaded,manifest=load_field(tmp_path/'isolated')
    actual=loaded.query(packet)
    assert manifest['external_scene_dependencies']==[]
    assert torch.equal(expected.selected,actual.selected)
    torch.testing.assert_close(expected.membership_scores,actual.membership_scores,rtol=0,atol=0)
    with (tmp_path/'isolated/field.pt').open('ab') as handle:handle.write(b'tampered')
    with pytest.raises(ValueError,match='digest'):load_field(tmp_path/'isolated')


def test_invalid_prompts_and_unsupported_elements_are_not_silently_relocated():
    field=example();field.observed[0]=False
    with pytest.raises(ValueError,match='observation support'):field.query(point_3d(field,field.xyz[0]))
    with pytest.raises(ValueError,match='tolerance'):point_3d(field,[100.,0.,0.])
    with pytest.raises(ValueError,match='contradictory'):
        field.query(QueryPacket('point_3d',positive=Support(torch.tensor([2]),torch.ones(1)),
                        negative=Support(torch.tensor([2]),torch.ones(1))))


def test_complete_category_menu_and_query_batches_produce_same_scores():
    field=example();field.observed[0]=False
    queries=torch.randn(4,9)
    scores,labels=field.category_partition(queries)
    torch.testing.assert_close(scores,torch.cat([field.semantic.score(q) for q in queries],dim=1))
    assert labels[0]==-1 and torch.equal(labels[1:],scores[1:].argmax(1))


def test_partly_unobserved_ray_keeps_coverage_and_signed_rays_remain_uncertain():
    field=example();field.observed[0]=False
    ray=Support(torch.tensor([0,1]),torch.tensor([.7,.3]))
    negative=Support(torch.tensor([1,2]),torch.tensor([.1,.9]))
    result=field.query(QueryPacket('point_2d',positive=ray,negative=negative))
    assert result.diagnostics['observed_positive_mass_fraction']==pytest.approx(.3)
    assert result.membership_scores[0]==0
    assert result.candidate_indices.tolist()==[0,1]
    assert not result.diagnostics['hard_positive_constraint']


def test_true_signed_elements_are_constraints_only_at_positive_weight_support():
    field=example()
    packet=QueryPacket('point_3d',exact_positive=True,
        positive=Support(torch.tensor([0,1]),torch.tensor([1.,0.])),
        negative=Support(torch.tensor([2,3]),torch.tensor([1.,0.])))
    result=field.query(packet)
    assert result.membership_scores[0]==1 and result.membership_scores[2]==0
    assert result.membership_scores[1]<1 and result.membership_scores[3]>0


def test_development_failures_remain_in_complete_roster():
    from radio_gs.query_field.evaluate import _summarize, choose_pixel
    rows=[dict(mode='point_2d',scale_condition='default',status='ok',projected_membership_iou=.6,actual_subset_iou=.4),
          dict(mode='point_2d',scale_condition='default',status='unsupported',projected_membership_iou=0.,actual_subset_iou=0.)]
    report=_summarize(rows)['point_2d/default']
    assert report['cases']==2 and report['failed']==1
    assert report['mean_projected_membership_iou']==.3 and report['mean_actual_subset_iou']==.2
    mask=torch.zeros(7,7,dtype=torch.bool);mask[1:6,1:6]=True
    assert choose_pixel(mask)==24
