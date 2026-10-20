"""Actual pre-response inputs for a matched reconstruction-stage experiment.

The retained native logits/identities are explicit frozen providers. Admission,
pixel phases, negative reliability, both category teachers and set heads are
preserved. These are stage interventions, not fresh cold-system runs.
"""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .response_study_adapters import SYSTEM, load, read
from .response_reconstruction import AdmittedResponseSystem
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions


def geometry_and_cameras(scene, binding):
    geo = FrozenGaussianGeometry.load(binding['path'], expected_sha256=binding['sha256'], device='cuda')
    return geo, MogeGeometryPrior(scene['camera_report'], scene['camera_report_sha256'])


@torch.no_grad()
def lerf_problem(job, data):
    from scripts.research.fit_consolidated_response import iter_source_response_views
    fit_path = SYSTEM/'lerf'/job['name']/'query/text/response/fit.json'
    fit = read(fit_path); inputs = read(fit['inputs']); scene = read(inputs['scene'])
    if fit['solver'] != 'backprojection' or fit['finite_evidence'] or scene.get('instance_evidence_reliability'):
        raise ValueError('unexpected retained LERF contract')
    initial = load(fit['readout'])
    if initial['texts'] != data['texts'] or initial['semantic_geometry'] != data['semantic_geometry']:
        raise ValueError('pre-fusion response query/carrier changed')
    geo, gp = geometry_and_cameras(scene, initial['semantic_geometry'])
    prior = initial['scores'].cuda(); observed = initial['semantic_observed'].cuda()
    observations = []
    for op, target, pos, neg, native, confidence in iter_source_response_views(
            inputs, scene, initial, geo, gp, observed, prior):
        if op.shape[0] != 192*256//4:raise ValueError('original pixel phases changed')
        observations.append((op, target, (pos | (neg & ~native)).float()*confidence[None], 1.))
    paths = [fit_path, fit['inputs'], inputs['scene'], fit['readout'],
             scene['camera_report'], initial['semantic_geometry']['path']]
    paths += [Path(inputs['detections'])/f'identity_{f}.pt' for f in scene['source_frames']]
    return AdmittedResponseSystem(prior, observations), dict(
        retained_solver='diagonal',views=len(observations),native_positive_veto=True,
        original_pixel_phase=True,all_identity_and_set_providers_preserved=True), paths


@torch.no_grad()
def nvos_problem(job, data):
    fit_path = SYSTEM/'nvos'/job['name']/'query/reference/0/fit.json'
    fit = read(fit_path); scene = read(fit['scene'])
    if (fit['explicit_evidence'] or fit['negative_policy'] != 'original'
            or fit['positive_core'] != 'eroded' or fit['reference_only']
            or scene.get('instance_evidence_reliability') or fit['steps'] != 120):
        raise ValueError('unexpected retained NVOS contract')
    tracks = load(fit['tracks']); binding = data['semantic_geometry']
    geo, gp = geometry_and_cameras(scene, binding)
    h,w = scene['source_fit_hw']; radius = scene['boundary_radius']; n = len(geo.xyz)
    observed = torch.zeros(n,device='cuda',dtype=torch.bool); observations=[]
    for frame in scene['source_frames']:
        cam=gp.cameras[frame]; k=cam.intrinsic.cuda().clone()
        k[0]*=w/cam.width; k[1]*=h/cam.height
        renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,
            near_plane=.001*gp.scale,far_plane=100*gp.scale)
        hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),
            height=h,width=w,include_depths=False)
        pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda()
        weight=hit['weights'].float().cuda();observed[gid[weight>1e-8]]=True
        total=torch.zeros(h*w,device='cuda').scatter_add_(0,pid,weight)
        record=tracks['frames'].get(frame)
        if record is None or record['object_logit'] <= 0:continue
        op=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/total[pid].clamp_min(1e-8),(h*w,n)).coalesce()
        target=F.interpolate(record['logits'].float()[None,None].cuda(),size=(h,w),
            mode='bilinear',align_corners=False).sigmoid()
        binary=(target>.5).float()
        pos=(-F.max_pool2d(-binary,2*radius+1,1,radius)>.5).flatten()[:,None]
        neg=(F.max_pool2d(binary,2*radius+1,1,radius)<.5).flatten()[:,None]
        known=(pos.float()+neg.float()*float(record.get('negative_reliability',1.)))*(total[:,None]>1e-8)
        observations.append((op,target.flatten()[:,None],known,1.))
    if not torch.equal(observed.cpu(),data['semantic_observed']):
        raise ValueError('reference reconstruction observed carrier changed')
    paths=[fit_path,fit['scene'],fit['query'],fit['tracks'],binding['path'],scene['camera_report']]
    return AdmittedResponseSystem(torch.full((n,1),.5,device='cuda'),observations), dict(
        retained_solver='rendered_huber',views=len(observations),negative_reliability_preserved=True,
        neutral_response_initialization=.5,independent_neutral_set_initialization_is_not_used=True),paths


@torch.no_grad()
def scannet_problem(job, data):
    from radio_gs.models.native_query_encoders import encode_category_text
    from .native_observation_prior import compiled_observation_prior
    from .local_element_fusion import compile_local_evidence
    scene=read(job['scene']);aux_path=scene['local_auxiliary_scene'];aux=read(aux_path)
    if scene.get('local_element_checkpoint'):raise ValueError('learned fusion is not this baseline')
    aux['native_observation_reliability']=scene['native_observation_reliability']
    aux.pop('observation_weighting',None)
    initial=data['native_logits'].cuda();observed=data['semantic_observed'].cuda()
    query=encode_category_text(aux,data['texts'],device='cuda')
    auxiliary,_=compiled_observation_prior(aux,query,observed,initial)
    compiled=compile_local_evidence(initial,observed,scene,data['texts'],auxiliary,
                                    capture_observations=True)
    legacy=compiled['average_logits'].sigmoid()
    error=float((legacy-data['category_logits'].cuda().sigmoid()).abs().max())
    if error>2e-6:raise AssertionError(f'two-native production replay differs: {error}')
    r=compiled['reconstruction']
    problem=AdmittedResponseSystem(r['initial'],r['observations'],prior=r['prior'],strength=r['strength'])
    paths=[job['scene'],aux_path,aux['native_observation_field']]
    if aux.get('category_query_encoder',{}).get('checkpoint'):
        paths.append(aux['category_query_encoder']['checkpoint'])
    paths += [v[k] for v in scene['source_evidence']['views'] for k in ['identity','contributions']]
    return problem,dict(retained_solver='diagonal',views=len(r['observations']),
        two_native_teachers_preserved=True,native_prior_strength_per_visibility=.1,
        production_replay_max_error=error,complete_category_menu=True,
        frozen_auxiliary_text_encoder_replayed=True),paths


def finish_response(job, data, problem, value):
    if job['benchmark']=='scannet':
        # Match the production logit round-trip and unsupported first-prior rule.
        return torch.where(problem.mass>1e-8,value.clamp(1e-6,1-1e-6),problem.initial)
    value=value.clone();value[~data['semantic_observed'].cuda()]=0
    if job['benchmark']=='nvos':value[problem.mass<=1e-8]=0
    return value
