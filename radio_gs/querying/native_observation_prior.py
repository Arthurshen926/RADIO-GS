"""Query-conditioned response mixture over query-independent native regions.

This is a categorical readout of preserved observations, not an object inventory
or a learned compact student. The compiled path reads only native descriptors
and sparse element weights; the reference path additionally loads masks and
camera/geometry state. Complete category competition precedes view aggregation.
"""
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from .observation_readout import native_identity, project_observations, corroborate_identity


@torch.inference_mode()
def compiled_observation_prior(scene, query, observed, fallback_logits, temperature=100., field=None):
    start = time.monotonic()
    supplied_by_caller = field is not None
    path = Path(scene['native_observation_field'])
    if field is None:
        field = torch.load(path, map_location='cpu', weights_only=False)
    if scene.get('native_observation_provenance'):
        from radio_gs.evidence_core.provenance import attach_provenance
        field = attach_provenance(field, path, scene['native_observation_provenance'],
                                  source_frames=scene.get('source_frames'))
    descriptor = F.normalize(field['descriptors'].float().to(query.device), dim=-1)
    identity=native_identity(descriptor,query)
    if 'member_identity_descriptors' in field:
        member=F.normalize(field['member_identity_descriptors'].float().to(query.device),dim=-1)
        identity=corroborate_identity(identity,native_identity(member,query),field['member_identity_coverage'].to(query.device))
    response = torch.softmax(temperature * identity, dim=-1)
    from radio_gs import evidence_core
    if evidence_core.enabled():
        from radio_gs.evidence_core.store import CompiledRegionEvidence
        transfer = CompiledRegionEvidence.from_payload(field).transfer(query.device)
    else:
        transfer = torch.sparse_csr_tensor(field['crow'].to(query.device), field['columns'].to(query.device),
                                         field['weights'].to(query.device), size=field['shape'])
    probability = project_observations(transfer, response)
    reliability_report = None
    if scene.get('native_observation_reliability'):
        from radio_gs.querying.observation_reliability import reliable_observation_response
        probability, reliability_report = reliable_observation_response(
            transfer, response, probability, field, scene['native_observation_reliability'])
    finite = probability.clamp(torch.finfo(probability.dtype).tiny, 1-torch.finfo(probability.dtype).eps)
    supported = field['supported'].to(query.device)
    logits = torch.where(supported[:, None], torch.logit(finite), fallback_logits)
    return logits, dict(kind='compiled_region_response_mixture', regions=len(descriptor),
        support_difference=int((supported != observed).sum()), temperature=temperature,
        observation_field_bytes=path.stat().st_size, source_rgb_read=False, source_masks_read=False,
        cameras_read=False, historical_query_answers_read=False, learned_compact_field=False,
        query_independent_compilation=True, seconds=time.monotonic()-start,
        field_deserialization_included=not supplied_by_caller, reliability=reliability_report,
        member_identity=field.get('member_identity_build'),
        region_provenance=field.get('region_provenance'))


@torch.inference_mode()
def native_observation_prior(scene, query, observed, fallback_logits, temperature=100.):
    start=time.monotonic();device=query.device
    geometry=FrozenGaussianGeometry.load(scene['geometry']['path'],expected_sha256=scene['geometry']['sha256'],device=device)
    cameras={c['frame_index']:c for c in json.loads(Path(scene['source_cameras']).read_text())['records']}
    total=torch.zeros_like(fallback_logits)
    mass=torch.zeros(len(observed),1,device=device)
    region_count=0;asset_bytes=0
    for frame in scene['source_frames']:
        camera=cameras[frame]
        path=Path(scene['native_region_directories'][0])/f'frame_{frame:05d}.pt'
        cache=torch.load(path,map_location='cpu',weights_only=False)
        sam_path=Path(cache['sam_path']);sam=torch.load(sam_path,map_location='cpu',weights_only=False)
        asset_bytes+=path.stat().st_size+sam_path.stat().st_size
        masks=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])[cache['native_proposal_indices'].numpy()].copy()
        if not len(masks):continue
        image_h,image_w=cache['native_mask_hw'];h,w=round(image_h*256/image_w),256
        membership=F.interpolate(torch.as_tensor(masks).float()[:,None],size=(h,w),mode='nearest-exact')[:,0].flatten(1).to(device)
        count=membership.sum(0)
        descriptor=F.normalize(cache['descriptors'].half().float().to(device),dim=-1)
        response=torch.softmax(temperature*native_identity(descriptor,query),dim=-1)
        target=membership.T@response/count[:,None].clamp_min(1)
        k=torch.as_tensor(camera['intrinsic'],device=device,dtype=torch.float32).clone()
        k[0]*=w/camera['width'];k[1]*=h/camera['height']
        renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001,far_plane=100.)
        hit=rasterize_single_view_contributions(geometry,renderer,torch.as_tensor(camera['world_to_camera'],device=device,dtype=torch.float32),height=h,width=w,include_depths=False)
        pixel,element=hit['pixel_ids'].long(),hit['gaussian_ids'].long()
        weight=hit['weights'].float()*(count[pixel]>0)
        adjoint=torch.sparse_coo_tensor(torch.stack([element,pixel]),weight,(len(observed),h*w)).coalesce()
        total+=torch.sparse.mm(adjoint,target)
        mass+=torch.sparse.mm(adjoint,torch.ones(h*w,1,device=device))
        region_count+=len(descriptor)
    probability=total/mass.clamp_min(1e-8)
    # Preserve low-probability class order, including evaluation on submenus.
    finite=probability.clamp(torch.finfo(probability.dtype).tiny,1-torch.finfo(probability.dtype).eps)
    logits=torch.where(mass>0,torch.logit(finite),fallback_logits)
    return logits,dict(kind='region_response_mixture',frames=len(scene['source_frames']),regions=region_count,
        support_difference=int(((mass[:,0]>0)!=observed).sum()),temperature=temperature,
        region_asset_bytes=asset_bytes,asset_cost_scope='native region and automatic mask files; native field, geometry, cameras, encoders and runtime grounding caches are additional',
        source_rgb_read=False,historical_query_answers_read=False,learned_compact_field=False,
        query_conditioned_response=True,seconds=time.monotonic()-start)
