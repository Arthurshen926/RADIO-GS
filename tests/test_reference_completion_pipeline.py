import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from radio_gs.pipeline.reference_completion_contract import PERMISSIONS, validate_profile, validate_geometry_training, sha
from radio_gs.pipeline.runtime import PipelineConfig, build_plan
from radio_gs.querying.image_response_geometry import normalized_image_operator


def fixture(tmp_path):
    def asset(name,data=b'authorized input'):
        path=tmp_path/name;path.write_bytes(data)
        return dict(path=str(path),sha256=sha(path))
    profile=dict(schema='radiogs.signed_reference_completion.v1',policy='quality',native_wh=[1008,756],
        solver_steps=120,threshold=.5,footprint_backend='gsplat_cuda',points_per_sign=16,gpu_memory_fraction=.38,
        target_frame=1,permissions=dict(PERMISSIONS),dino_repository='/installed/dino',
        image_checkpoint=asset('sam.pt'),dino_checkpoint=asset('dino.pt'),
        geometry_fit=asset('fit.json',b'{"source_frames":[0],"validation_frames":[]}'),
        inputs={k:asset(k+'.png') for k in ['reference_rgb','reference_mask','positive_scribble','negative_scribble','target_rgb']})
    scene=dict(scene='fixture',geometry=dict(path=str(tmp_path/'geometry.pt'),sha256='a'*64),source_frames=[0],tracking_frames=[0])
    query=dict(id='q',intent='reference_instance',modality='prompt_2d',reference_frame=0,
        reference_mask_is_gt=False,prompt_raster_wh=[1008,756],
        **{k:profile['inputs'][k]['path'] for k in ['reference_rgb','reference_mask','positive_scribble']})
    for name,value in [('profile',profile),('scene',scene),('query',query)]:
        (tmp_path/(name+'.json')).write_text(json.dumps(value))
    return profile,scene,query


@pytest.mark.parametrize('change',['permission','target_source','reference_gt','extra_gt','point_budget','asset_path','hash'])
def test_explicit_reference_contract_rejects_permission_and_binding_changes(tmp_path,change):
    p,s,q=fixture(tmp_path)
    assert validate_profile(p,s,q,verify_assets=True)==PERMISSIONS
    if change=='permission':p['permissions']['target_rgb']=False
    if change=='target_source':p['target_frame']=0
    if change=='reference_gt':q['reference_mask_is_gt']=True
    if change=='extra_gt':p['target_gt']='/hidden/truth.png'
    if change=='point_budget':p['points_per_sign']=32
    if change=='asset_path':p['inputs']['reference_rgb']['path']='/other/frame.jpg'
    if change=='hash':Path(p['inputs']['target_rgb']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):validate_profile(p,s,q,verify_assets=True)


def test_geometry_training_cannot_borrow_target_rgb(tmp_path):
    p,_,_=fixture(tmp_path)
    validate_geometry_training(p,dict(source_frames=[0],validation_frames=[]))
    with pytest.raises(ValueError,match='target RGB'):
        validate_geometry_training(p,dict(source_frames=[0],validation_frames=[1]))


def test_plan_builds_fresh_initializer_then_completion_and_retains_default(tmp_path):
    fixture(tmp_path)
    options=dict(scene=str(tmp_path/'scene.json'),queries=str(tmp_path/'query.json'),output=str(tmp_path/'run'),reference_tracks='/declared/tracks.pt')
    base=build_plan(PipelineConfig(**options))
    new=build_plan(PipelineConfig(**options,reference_completion=str(tmp_path/'profile.json')))
    assert base['target_rgb_allowed'] is False and new['target_rgb_allowed'] is True
    assert base['commands'][:-1]==new['commands'][:-2]
    assert new['commands'][-2][-1].endswith('/base_readout.pt')
    assert new['commands'][-1][2]=='radio_gs.pipeline.reference_completion'
    assert new['reference_completion_permissions']['negative_scribble']
    assert not new['inherited_query_predictions']
    with pytest.raises(ValueError,match='source-assisted'):
        build_plan(PipelineConfig(**options,reference_completion=str(tmp_path/'profile.json'),access='map-only'))


def test_native_operator_matches_historical_formula_with_unseen_pixels():
    h=dict(pixel_ids=torch.tensor([0,0,1,1,2]),gaussian_ids=torch.tensor([0,1,0,2,1]),
        weights=torch.tensor([.6,.2,.3,.4,.5]),rendered_alpha=torch.tensor([.8,.7,.5,0.]))
    observed=torch.tensor([True,False,True]);p,g=h['pixel_ids'],h['gaussian_ids']
    weights=h['weights']*observed[g]
    mass=torch.zeros(4).scatter_add_(0,p,weights)
    valid=(mass/h['rendered_alpha'].clamp_min(1e-8)>.5)&(mass>1e-8)
    expected=torch.sparse_coo_tensor(torch.stack([p,g]),weights/mass[p].clamp_min(1e-8),(4,3)).coalesce()
    op,mask=normalized_image_operator(h,observed,4)
    assert torch.equal(expected.indices(),op.indices()) and torch.equal(expected.values(),op.values())
    assert torch.equal(valid[:,None],mask) and not mask[2:].any()


def test_complete_readout_uses_same_solve_and_keeps_independent_set(monkeypatch):
    from radio_gs.pipeline.reference_completion import completed_readout
    from radio_gs.querying.image_conditioned_response import lift_image_response
    initial=dict(scores=torch.tensor([[.2],[0.],[.8]]),semantic_observed=torch.tensor([True,False,True]),
        selection_scores=torch.tensor([[.9],[0.],[.1]]),texts=['q'])
    op=torch.sparse_coo_tensor(torch.tensor([[0,2],[0,2]]),torch.ones(2),(3,3)).coalesce()
    valid=torch.tensor([[True],[False],[True]]);image=torch.tensor([[.9],[.7],[.1]])
    worker=dict(choices=dict(feasible='semantic_memory',quality='constrained_memory'),
        decoder=dict(semantic_memory=dict(positive_coverage=1.,negative_violation=0.,predicted_quality=.7),
                     constrained_memory=dict(predicted_quality=.8)),
        target_gt_available=False,reference_gt_available=False,negative_scribble_used=True,sam_eval=True,dino_eval=True)
    p=dict(policy='quality',solver_steps=120,permissions=PERMISSIONS,target_frame=1,native_wh=[1,3],threshold=.5)
    monkeypatch.setenv('RADIO_GS_EVIDENCE_CORE','legacy')
    old,_=lift_image_response(initial['scores'],op,image,(2*image-1).abs()*valid,120)
    monkeypatch.setenv('RADIO_GS_EVIDENCE_CORE','shared_v1')
    result,report=completed_readout(initial,op,image,valid,p,worker)
    assert torch.equal(result['scores'],old) and not torch.equal(result['scores'],initial['scores'])
    assert result['selection_scores'] is initial['selection_scores']
    assert result['reference_completion']['selected_reference']=='constrained_memory'
    assert report['unsupported_changed']==0 and result['scores'][1]==0


def function(path,name):
    module=ast.parse(Path(path).read_text())
    return next(n for n in module.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)


def test_migrated_worker_and_rgb_transform_preserve_historical_function_ast():
    old=function('scripts/research/decode_tmm_nvos_reference_admission.py','main')
    new=function('radio_gs/pipeline/reference_completion_worker.py','main')
    assert ast.dump(old)==ast.dump(new)
    old=function('scripts/research/decode_tmm_nvos_memory_response.py','LazyImages')
    new=function('radio_gs/querying/reference_transport.py','LazyImages')
    assert ast.dump(old)==ast.dump(new)
    for name in ['camera_args','render_responses']:
        old=function('scripts/research/run_tmm_nvos_image_response.py',name)
        new=function('radio_gs/querying/image_response_geometry.py',name)
        new.body=[n for n in new.body if not isinstance(n,ast.ImportFrom)]
        assert ast.dump(old)==ast.dump(new)


def test_nvos_scoring_discloses_rgb_and_keeps_gray_gt_conventions():
    from radio_gs.evaluation.reference_response import readout_permissions,score_response_tracks
    from sklearn.metrics import average_precision_score
    metadata=dict(permissions=PERMISSIONS,target_frame=1,native_wh=[3,2],threshold=.5,
                  selection_profile='inherited_source_only_selected_set_unchanged')
    evaluation=dict(target_frame=1,raster_wh=[3,2],threshold=.5)
    d=dict(reference_completion=metadata)
    assert readout_permissions(d,evaluation)['target_rgb']
    assert not readout_permissions(d,evaluation,selected_subset=True)['target_rgb']
    assert not readout_permissions({},evaluation)['negative_scribble']
    x=np.array([[.5,.6,.2],[.9,.5,.1]],np.float32);g=np.array([[255,128,0],[255,0,0]],np.uint8)
    rows=score_response_tracks(x,g)
    assert rows['retained_gt_gt127']['tp']==3 and rows['ludvig_gt_maximum']['tp']==2
    for track,truth in [('retained_gt_gt127',g>127),('ludvig_gt_maximum',(g//g.max()).astype(bool))]:
        assert rows[track]['average_precision']==pytest.approx(average_precision_score(truth.ravel(),x.ravel()))
    with pytest.raises(ValueError,match='contract differs'):
        readout_permissions(d,dict(evaluation,target_frame=3))


def test_extension_recipe_preserves_other_tasks_and_one_policy_for_all_references(tmp_path):
    from scripts.research.run_shared_evidence_migration import validate_extension_recipe
    base=dict(jobs=[dict(benchmark='nvos',name=n,scene='scene.json') for n in ['a','b']]+[
        dict(benchmark='uqis',name='room',scene='map.json')])
    recipe=copy.deepcopy(base)
    for job in recipe['jobs'][:2]:
        path=tmp_path/(job['name']+'.json');path.write_text(json.dumps(dict(policy='quality')))
        job['reference_completion']=str(path);job['prompt_contract']='signed';job['input_track']='target_rgb'
    validate_extension_recipe(recipe,base)
    recipe['jobs'][-1]['scene']='different.json'
    with pytest.raises(ValueError,match='other benchmarks'):validate_extension_recipe(recipe,base)
    recipe['jobs'][-1]=base['jobs'][-1]
    (tmp_path/'b.json').write_text(json.dumps(dict(policy='feasible')))
    with pytest.raises(ValueError,match='task-wise'):validate_extension_recipe(recipe,base)
