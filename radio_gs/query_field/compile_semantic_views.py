"""Compile legal source-region functions onto frozen Gaussian rows.

Only construction reads masks/cameras/RGB hashes. Replay contracts pixel-region
responses before projection, independently of exported Gaussian-mode products.
"""
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F
from .geometry import projection
from .semantic_measure import SemanticMeasure
from .storage import load_field,save_field,sha256
from .observations import source_sam_metadata


@torch.no_grad()
def compile_scene(job,output):
    from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
    from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
    from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
    started=time.monotonic();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    scene=json.loads(Path(job['source_scene']).read_text())
    authority=json.loads(Path(scene['source_rgb_authority']).read_text())
    if authority['information_policy'].get('query_text_used') or authority['information_policy'].get('benchmark_ground_truth_used') or authority['information_policy'].get('target_or_evaluation_rgb_used'):
        raise ValueError('illegal source authority')
    images={int(r['image_id'].split('_')[-1]):r for r in authority['images']}
    frames=job['frames']
    if set(frames)-set(scene['source_frames']) or set(frames)&set(job['excluded_frames']):raise ValueError('illegal frame roster')
    geometry=FrozenGaussianGeometry.load(scene['geometry']['path'],expected_sha256=scene['geometry']['sha256'],device='cuda')
    cameras=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256'])
    parent,manifest=load_field(job['parent'])
    for k in ['xyz','normals','scales','opacities','quaternions']:
        if not torch.equal(getattr(parent,k),getattr(geometry,k).cpu()):raise ValueError('carrier differs')
    count=len(parent.xyz);total=torch.zeros(count,1,device='cuda');direct=torch.zeros(count,3,device='cuda')
    blocks=[];descriptors=[];receipts=[];queries=None;negatives=None
    for frame in frames:
        path=next(Path(p)/f'frame_{frame:05d}.pt' for p in scene['native_region_directories'] if (Path(p)/f'frame_{frame:05d}.pt').exists())
        region=torch.load(path,map_location='cpu',weights_only=False)
        if region.get('query_independent') is not True or region['frame']!=frame or region['encoder']['path']!=scene['text_encoder']:
            raise ValueError('native source header differs')
        sam_path=Path(region['sam_path']);sam=torch.load(sam_path,map_location='cpu',weights_only=False)
        source_hash,_=source_sam_metadata(sam['metadata']);rgb=images[frame]
        if source_hash!=sha256(rgb['path']) or source_hash!=rgb['sha256']:raise ValueError('source mask RGB binding differs')
        masks=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])[region['native_proposal_indices'].numpy()].copy()
        if list(masks.shape[1:])!=list(region['native_mask_hw']):raise ValueError('native raster differs')
        h,w=scene['source_fit_hw'];m=F.interpolate(torch.from_numpy(masks).float()[:,None],(h,w),mode='nearest-exact')[:,0].flatten(1).cuda()
        cover=m.sum(0);descriptor=region['descriptors'].half().float().cuda()
        if not len(descriptor) or len(descriptor)!=len(m):raise ValueError('empty or unaligned source regions')
        if queries is None:
            if len(descriptor)<7:raise ValueError('first source needs seven fixed replay directions')
            indices=np.linspace(0,len(descriptor)-1,7).round().astype(int)
            probes=F.normalize(descriptor[indices],dim=1);queries=probes[:3];negatives=probes[3:]
        raw,_,_,_=projection(geometry,cameras.cameras[frame],h,w,cameras.scale)
        pixel,element=raw.indices();weight=raw.values()*(cover[pixel]>0)
        adjoint=torch.sparse_coo_tensor(torch.stack([element,pixel]),weight,(count,h*w)).coalesce()
        region_pixels=m.T/cover[:,None].clamp_min(1)
        transfer=torch.sparse.mm(adjoint,region_pixels)
        rows,cols=torch.where(transfer>0)
        blocks.append(sparse.csr_matrix((transfer[rows,cols].cpu().numpy(),(rows.cpu().numpy(),cols.cpu().numpy())),shape=(count,len(m))))
        descriptors.append(region['descriptors'].half())
        mass=torch.sparse.mm(adjoint,torch.ones(h*w,1,device='cuda'));total+=mass
        direction=F.normalize(descriptor,dim=1)
        response=(10*(direction@queries.T-(direction@negatives.T).max(1).values[:,None])).sigmoid()
        direct+=torch.sparse.mm(adjoint,region_pixels@response)
        receipts.append(dict(frame=frame,regions=len(m),native=dict(path=str(path),sha256=sha256(path)),
            sam=dict(path=str(sam_path),sha256=sha256(sam_path)),rgb=dict(path=rgb['path'],sha256=source_hash)))
        print('source',job['name'],frame,len(m),flush=True)
    a=sparse.hstack(blocks,format='csr');del blocks
    mass=total[:,0].cpu().numpy();a=(sparse.diags(1/np.maximum(mass,np.finfo(np.float32).tiny))@a).tocsr()
    a.eliminate_zeros();used=np.unique(a.indices);a=a[:,used].tocsr();a.sort_indices()
    # Preserve numerical raw projection mass; no new equal-view reweighting.
    atoms=torch.cat(descriptors)[used].clone();supported=np.asarray(a.getnnz(1)>0)
    state=dict(descriptors=atoms,pointer=torch.from_numpy(a.indptr.astype(np.int64)),index=torch.from_numpy(a.indices.astype(np.int64)),
        mass=torch.from_numpy(a.data.astype(np.float32)),supported=torch.from_numpy(supported))
    measure=SemanticMeasure(state,count).cuda();expected=direct/total.clamp_min(torch.finfo(total.dtype).tiny)
    replay=[]
    for i,q in enumerate(queries):
        actual=measure.union_probabilities(q,negatives,torch.zeros(count,device='cuda'))
        error=float((actual-expected[:,i]).abs().max());different=int(((actual>.5)!=(expected[:,i]>.5)).sum())
        if error>2e-5:raise ValueError('source pixel->field union replay failed: '+str(error))
        replay.append(dict(query=i,maximum_error=error,selection_differences=different))
    before={k:v.clone() for k,v in parent.state_dict().items()}
    parent.semantic_measure=measure.cpu();parent.semantic_measure_rule='measure'
    if not all(torch.equal(parent.state_dict()[k],v) for k,v in before.items()):raise ValueError('parent changed')
    exported=save_field(parent,out/'field',dict(parent_field=manifest['files'],source_scene=dict(path=job['source_scene'],sha256=sha256(job['source_scene'])),
        frames=frames,excluded_frames=job['excluded_frames'],sources=receipts,training_steps=0,
        source_queries_used=False,target_labels_used=False,legacy_calibration_reused=False,
        operator='normalized rendered contribution to overlapping query-free source regions',uncompressed_measure=True,
        primary_query='text_union measure with common member corroboration'))
    report=dict(scene=job['name'],frames=len(frames),atoms=len(atoms),nonzeros=a.nnz,supported=int(supported.sum()),
        parent_observed=int(parent.observed.sum()),tensor_bytes=exported['tensor_bytes'],parent_tensor_bytes=manifest['tensor_bytes'],
        source_replay=replay,parent_unchanged=True,source_receipts=receipts,field=exported['files'],seconds=time.monotonic()-started)
    (out/'construction.json').write_text(json.dumps(report,indent=2)+'\n');return report
