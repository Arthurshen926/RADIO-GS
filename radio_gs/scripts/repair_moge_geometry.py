"""Geometry-only source audit, joint scale fit, and frozen Gaussian rebuilding.

No features, semantics, benchmark queries or masks are read. Reference depth
is independent-view MoGe, not ground-truth geometry; camera calibration is
shared source COLMAP authority. Output geometry must pass checks before use.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.joint_scale import fit_joint_depth_scales, recover_unanchored_view_scales
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.geometry.ray_alignment import align_normals_to_intrinsics
from radio_gs.geometry.affine_depth_normals import affine_depth_normals
from radio_gs.v4.carrier import Camera
from radio_gs.v4.geometry.tsdf_fusion import DepthObservation, SparseSurfaceFusion
from radio_gs.utils.immutable_artifacts import sha256_file


def load_views(prior, width, device, allow_global_scale_fallback=False, *, legacy_affine_normals=False):
    views=[]
    for frame in sorted(prior.records):
        row=prior.records[frame]
        if sha256_file(row['prediction']) != row['prediction_sha256']:
            raise ValueError('prediction bytes changed')
        cal=prior.calibration[frame]
        if cal.get('source_geometry_override') is not None:
            if not cal['accepted']:continue
            from radio_gs.geometry.source_geometry_override import load_source_geometry_override
            native=prior.cameras[frame]
            materialized=load_source_geometry_override(cal['source_geometry_override'],frame=frame,prediction_sha256=row['prediction_sha256'],camera=native)
            depth=materialized['depth'];h,w=depth.shape
            if w!=width:raise ValueError('local geometry materialization width differs; rebuild at requested grid')
            k=materialized['intrinsic'].to(device);pose=native.camera_to_world.float().to(device)
            y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
            rays=torch.stack([x+.5,y+.5,torch.ones_like(x)],dim=-1).float()@torch.linalg.inv(k).T
            views.append(dict(frame=frame,local_calibration_accepted=True,depth=depth.to(device),valid=materialized['valid'].to(device),normals=materialized['normals'].to(device),k=k,pose=pose,rays=rays))
            continue
        payload=torch.load(row['prediction'],map_location='cpu',weights_only=False)
        if payload['model_checkpoint_sha256'] != prior.manifest['model_checkpoint_sha256']:
            raise ValueError('prediction model changed')
        cal=prior.calibration[frame]
        if not cal['accepted'] and not allow_global_scale_fallback:
            continue
        local_scale=cal['scale'] if cal['accepted'] else 1.0
        local_offset=cal['offset'] if cal['accepted'] else 0.0
        uncalibrated_depth=payload['point_map'][...,2].float()*prior.scale
        depth=uncalibrated_depth*local_scale+local_offset
        valid=payload['validity'].bool() & torch.isfinite(depth) & (depth>0)
        normals=payload['normals'].float()
        if payload.get('intrinsics') is not None:
            # Official MoGe caches carry normalized image intrinsics.
            source_k=payload['intrinsics'].float().clone()
            native=prior.cameras[frame]
            target_k=native.intrinsic.float().clone()
            target_k[0]/=native.width;target_k[1]/=native.height
            normals=align_normals_to_intrinsics(normals,source_k,target_k)
        if local_offset != 0 and not legacy_affine_normals:
            native=prior.cameras[frame]
            native_k=native.intrinsic.float().clone()
            native_k[0]/=native.width;native_k[1]/=native.height
            nh,nw=depth.shape
            ny,nx=torch.meshgrid(torch.arange(nh),torch.arange(nw),indexing='ij')
            native_rays=torch.stack([(nx+.5)/nw,(ny+.5)/nh,torch.ones_like(nx)],dim=-1).float()@torch.linalg.inv(native_k).T
            legal=valid&torch.isfinite(uncalibrated_depth)&(uncalibrated_depth>0)
            normals[legal]=affine_depth_normals(native_rays[legal]*uncalibrated_depth[legal][:,None],normals[legal],local_scale,local_offset)
        # Native model and COLMAP can have slightly different intrinsics;
        # mapping uses the calibrated source-camera rays, never cached x/y.
        h=max(1,round(depth.shape[0]*width/depth.shape[1]))
        depth=F.interpolate(depth[None,None],size=(h,width),mode='nearest-exact')[0,0].to(device)
        valid=F.interpolate(valid[None,None].float(),size=(h,width),mode='nearest-exact')[0,0].bool().to(device)
        normals=F.interpolate(normals.permute(2,0,1)[None],size=(h,width),mode='nearest-exact')[0].permute(1,2,0).to(device)
        native=prior.cameras[frame]
        k=native.intrinsic.float().to(device).clone();k[0]*=width/native.width;k[1]*=h/native.height
        pose=native.camera_to_world.float().to(device)
        y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(width,device=device),indexing='ij')
        rays=torch.stack([x+.5,y+.5,torch.ones_like(x)],dim=-1).float()@torch.linalg.inv(k).T
        views.append(dict(frame=frame,local_calibration_accepted=bool(cal['accepted']),depth=depth,valid=valid,normals=normals,k=k,pose=pose,rays=rays))
    return views


def correspondences(views, unit):
    records=[]
    # Temporal neighbors avoid unrelated surfaces; reciprocal depth agreement
    # and normal checks reject most occlusion/boundary correspondences.
    for i,src in enumerate(views):
        for j in range(i+1,min(len(views),i+3)):
            dst=views[j]
            valid=src['valid'].clone();valid[1::2]=False;valid[:,1::2]=False
            ys,xs=torch.where(valid)
            p=src['rays'][ys,xs]*src['depth'][ys,xs,None]
            pw=p@src['pose'][:3,:3].T+src['pose'][:3,3]
            qcam=(pw-dst['pose'][:3,3])@dst['pose'][:3,:3]
            uv=qcam@dst['k'].T
            x=(uv[:,0]/uv[:,2].clamp_min(1e-8)).floor().long()
            y=(uv[:,1]/uv[:,2].clamp_min(1e-8)).floor().long()
            h,w=dst['depth'].shape
            inside=(qcam[:,2]>0)&(x>=0)&(x<w)&(y>=0)&(y<h)
            x,y=x.clamp(0,w-1),y.clamp(0,h-1)
            d=dst['depth'][y,x]
            n1=src['normals'][ys,xs]@src['pose'][:3,:3].T
            n2=dst['normals'][y,x]@dst['pose'][:3,:3].T
            keep=inside&dst['valid'][y,x]&((qcam[:,2]-d).abs()<.03*unit+.08*d)&(F.cosine_similarity(n1,n2).abs()>.8)
            q=dst['rays'][y,x]*d[:,None]
            if int(keep.sum())<32:continue
            p,q=p[keep]@src['pose'][:3,:3].T,q[keep]@dst['pose'][:3,:3].T
            t=(src['pose'][:3,3]-dst['pose'][:3,3]).expand_as(p)
            count=p.shape[0]
            records.append((torch.full((count,),i,device=p.device),torch.full((count,),j,device=p.device),p,q,t))
    if not records:raise ValueError('no supported multiview correspondences')
    return tuple(torch.cat([r[k] for r in records]) for k in range(5))


def residual_summary(corr,scale):
    i,j,p,q,t=corr
    r=(scale[i,None]*p-scale[j,None]*q+t).norm(dim=1)/((p.norm(dim=1)+q.norm(dim=1))*.5).clamp_min(1e-8)
    return {'mean':float(r.mean()),'median':float(r.median()),'p90':float(r.quantile(.9))}


def build_geometry(views,scale,unit,voxel_metres,device,minimum_voxel_views=None):
    observations=[]
    for index,v in enumerate(views):
        h,w=v['depth'].shape
        camera=Camera(str(v['frame']),v['k'].cpu(),v['pose'].cpu(),h,w)
        observations.append(DepthObservation(camera,v['depth']*scale[index],v['valid'],v['valid'].float(),v['normals']))
    fused=SparseSurfaceFusion(voxel_metres*unit,minimum_views=2,device=device,
                             consistency_mode='source_depth_v2',relative_depth_tolerance=.01,minimum_voxel_views=minimum_voxel_views).fuse(observations)
    keep=fused.normals.norm(dim=1)>.5
    return FrozenGaussianGeometry(fused.centres[keep],fused.normals[keep],
                                  tangent_sigma=voxel_metres*unit*.75,
                                  normal_sigma=voxel_metres*unit*.1).to(device),fused


def evaluate_geometry(model,views,unit,prior=None,sparse_points=None):
    from gsplat import rasterization
    rows=[]
    for v in views:
        h,w=v['depth'].shape
        rendered,alpha,_=rasterization(means=model.get_xyz(),quats=model.get_rotation(),scales=model.get_scaling(),
            opacities=model.get_opacity().reshape(-1),colors=torch.zeros(model.get_xyz().shape[0],1,device=v['depth'].device),
            viewmats=torch.linalg.inv(v['pose'])[None],Ks=v['k'][None],width=w,height=h,
            near_plane=.001*unit,far_plane=100*unit,render_mode='RGB+ED',packed=False)
        depth=rendered[0,...,-1];a=alpha[0,...,0]
        valid=v['valid'];covered=valid&(a>.1)&torch.isfinite(depth)&(depth>0)
        ref=v['depth'][covered];pred=depth[covered]
        relative=(pred-ref).abs()/ref.clamp_min(1e-8)
        tolerance=.03*unit+.02*ref
        sparse_stats={}
        if prior is not None and sparse_points is not None:
            camera=prior.cameras[v['frame']]
            retained=[index for index,pid in enumerate(camera.point_ids.tolist()) if int(pid) in sparse_points]
            if retained:
                ids=torch.tensor(retained,dtype=torch.long)
                world=torch.stack([sparse_points[int(pid)] for pid in camera.point_ids[ids].tolist()]).to(depth.device)
                expected=((world-v['pose'][:3,3])@v['pose'][:3,:3])[:,2]
                xy=camera.xy[ids].to(depth.device)
                px=(xy[:,0]*w/camera.width).floor().long();py=(xy[:,1]*h/camera.height).floor().long()
                inside=(px>=0)&(px<w)&(py>=0)&(py<h)&(expected>0)
                px,py=px.clamp(0,w-1),py.clamp(0,h-1)
                observed=inside&covered[py,px]
                error=(depth[py[observed],px[observed]]-expected[observed]).abs()/expected[observed]
                prior_error=(v['depth'][py[observed],px[observed]]-expected[observed]).abs()/expected[observed]
                sparse_stats={'colmap_points_covered':int(observed.sum()),'colmap_points_available':int(inside.sum()),
                              'colmap_absrel':float(error.mean()) if error.numel() else None,
                              'colmap_median_absrel':float(error.median()) if error.numel() else None,
                              'colmap_agreement_coverage':float((error<.05).sum()/inside.sum().clamp_min(1)),
                              'moge_reference_colmap_absrel':float(prior_error.mean()) if error.numel() else None}
        rows.append({**sparse_stats,'frame':v['frame'],'coverage':float(covered.sum()/valid.sum().clamp_min(1)),
                     'absrel':float(relative.mean()) if ref.numel() else None,
                     'median_absrel':float(relative.median()) if ref.numel() else None,
                     'free_space_fraction':float((pred<ref-tolerance).float().mean()) if ref.numel() else None,
                     'depth_agreement_fraction':float(((pred-ref).abs()<=tolerance).float().mean()) if ref.numel() else None,
                     'agreement_coverage':float(((pred-ref).abs()<=tolerance).sum()/valid.sum().clamp_min(1)),
                     'reference_pixels':int(valid.sum()),'covered_pixels':int(covered.sum())})
    aggregate={key:sum(r[key] for r in rows if r[key] is not None)/max(1,sum(r[key] is not None for r in rows))
               for key in ('coverage','absrel','median_absrel','free_space_fraction','depth_agreement_fraction','agreement_coverage')}
    for key in ('colmap_absrel','moge_reference_colmap_absrel'):
        measured=[r for r in rows if r.get(key) is not None]
        if measured:
            total=sum(r['colmap_points_covered'] for r in measured)
            aggregate[key]=sum(r[key]*r['colmap_points_covered'] for r in measured)/total
    return {'aggregate':aggregate,'views':rows,'reference':'heldout_source_MoGe_and_source_only_COLMAP_sparse_points'}


def run(args):
    torch.set_num_threads(4)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    prior=MogeGeometryPrior(args.fusion_report,args.expected_report_sha256)
    device=torch.device(args.device)
    views=load_views(prior,args.width,device,args.allow_global_scale_fallback)
    if args.split_reference:
        split=json.loads(Path(args.split_reference).read_text())
        if split['source_authority_sha256'] != prior.source_authority_sha256:
            raise ValueError('split reference uses a different source frame authority')
        heldout_ids=set(split['heldout_frames'])
        heldout=[v for v in views if v['frame'] in heldout_ids]
        if len(heldout)!=len(heldout_ids):raise ValueError('missing fixed heldout views')
    else:
        heldout=views[::5]
        heldout_ids={v['frame'] for v in heldout}
    training=[v for v in views if v['frame'] not in heldout_ids][::args.mapping_stride]
    recovery=[]
    if args.recover_rejected_scales:
        training,recovery=recover_unanchored_view_scales(training)
    if len(training)<3 or not heldout:raise ValueError('insufficient source geometry split')
    corr=correspondences(training,prior.scale)
    mask=torch.arange(corr[0].numel(),device=device)%5==0
    train_corr=tuple(x[~mask] for x in corr);validation_corr=tuple(x[mask] for x in corr)
    ones=torch.ones(len(training),device=device)
    scale=fit_joint_depth_scales(*train_corr,num_views=len(training))
    before=residual_summary(validation_corr,ones);after=residual_summary(validation_corr,scale)
    accepted=after['mean']<before['mean'] and after['p90']<=before['p90']
    if not accepted:scale=ones
    report={'schema':'radio_gs.geometry_only_repair.v1','implementation_sha256':sha256_file(Path(__file__)),
            'affine_depth_normal_transport':'inverse_transpose_Xprime_equals_a_plus_b_over_Z_times_X_v1',
            'affine_depth_normal_implementation_sha256':sha256_file(Path(__file__).parents[1]/'geometry/affine_depth_normals.py'),
            'normal_ray_alignment':'inverse_transpose_intrinsics_v1','source_report':str(Path(args.fusion_report).resolve()),'source_report_sha256':sha256_file(args.fusion_report),
            'source_authority_sha256':prior.source_authority_sha256,
            'split_reference_sha256':sha256_file(args.split_reference) if args.split_reference else None,
            'mapping_frames':[v['frame'] for v in training], 'heldout_frames':[v['frame'] for v in heldout],
            'queries_opened':False,'semantic_features_opened':False,'geometry_trainable_during_semantics':False,
            'recovered_scales':recovery,
            'joint_scale':{'accepted':accepted,'before':before,'after_candidate':after,'scales':scale.tolist(),
                           'correspondences':corr[0].numel(),'validation_correspondences':int(mask.sum())},
            'width':args.width,'voxel_metres':args.voxel_metres,'mapping_stride':args.mapping_stride,
            'minimum_voxel_views':args.minimum_voxel_views,'allow_global_scale_fallback':args.allow_global_scale_fallback,'results':{}}
    evaluations=heldout[::max(1,len(heldout)//args.evaluation_views)][:args.evaluation_views]
    authority_path=next(r['path'] for r in prior.report['geometry_receipt']['inputs'] if r['role']=='source_authority')
    meta=json.loads(Path(authority_path).read_text())['metadata']
    report['historical_comparator_evaluated'] = not getattr(args, 'skip_historical_comparator', False)
    if report['historical_comparator_evaluated']:
        from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
        model,*rest=load_render_pipeline(meta['config'],meta['checkpoint'],device,strict_checkpoint_contract=True,load_ply_rgb_features=False)
        del rest
        report['results']['historical_gaussians']=evaluate_geometry(model,evaluations,prior.scale)
        del model;torch.cuda.empty_cache()
        print('historical geometry evaluated',flush=True)
    for name,scales in [('independent_scale',ones),('joint_scale',scale)]:
        model,fused=build_geometry(training,scales,prior.scale,args.voxel_metres,device,args.minimum_voxel_views)
        path=out/(name+'.pt')
        torch.save({'schema':'radio_gs.frozen_surface_gaussian_geometry.v1','state_dict':{k:v.cpu() for k,v in model.state_dict().items()},
                    'mapping_frames':report['mapping_frames'],'source_report_sha256':report['source_report_sha256'],
                    'source_authority_sha256':prior.source_authority_sha256,'scales':scales.cpu(),
                    'geometry_trainable_during_semantics':False},path)
        score=evaluate_geometry(model,evaluations,prior.scale)
        score.update(elements=int(model.get_xyz().shape[0]),geometry=str(path.resolve()),geometry_sha256=sha256_file(path))
        report['results'][name]=score
        print(name,score['aggregate'],flush=True)
        del model,fused;torch.cuda.empty_cache()
    from radio_gs.geometry.quality_gate import joint_scale_quality_gate
    report['joint_scale']['correspondence_accepted'] = report['joint_scale']['accepted']
    gate = joint_scale_quality_gate(report['results']['independent_scale']['aggregate'],
        report['results']['joint_scale']['aggregate'], correspondence_accepted=accepted)
    report['joint_scale']['rendered_quality_gate'] = gate
    report['joint_scale']['accepted'] = gate['accepted']
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--fusion-report',required=True);p.add_argument('--expected-report-sha256',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--width',type=int,default=320)
    p.add_argument('--voxel-metres',type=float,default=.015);p.add_argument('--evaluation-views',type=int,default=8)
    p.add_argument('--recover-rejected-scales',action='store_true')
    p.add_argument('--skip-historical-comparator',action='store_true',help='Build a new scene without requiring an unrelated historical Gaussian checkpoint; report comparator absence explicitly')
    p.add_argument('--split-reference',default='')
    p.add_argument('--allow-global-scale-fallback',action='store_true')
    p.add_argument('--mapping-stride',type=int,default=2);p.add_argument('--minimum-voxel-views',type=int,default=2)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    if args.width<16 or args.voxel_metres<=0 or args.evaluation_views<1 or args.mapping_stride<1 or args.minimum_voxel_views<1:p.error('invalid geometry dimensions')
    run(args)

if __name__=='__main__':main()
