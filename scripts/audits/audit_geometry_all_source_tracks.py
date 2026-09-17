"""Render frozen geometry against sparse tracks across mapping and heldout views.
No semantic images/masks; raw full-scene alpha compositor, not teacher gates.
"""
import argparse
import json
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points, _bilinear
from radio_gs.utils.immutable_artifacts import sha256_file


def stats(values):
    t=torch.tensor(values,dtype=torch.float64)
    return dict(count=len(t),median=float(t.median()),p90=float(t.quantile(.9)),mean=float(t.mean())) if len(t) else dict(count=0)


def run():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4)
    gp=Path('/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914')/a.scene/'report.json'
    report=json.loads(gp.read_text());prior=MogeGeometryPrior(report['source_report'],report['source_report_sha256'])
    g=report['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device)
    inputs={r['role']:r for r in prior.report['geometry_receipt']['inputs']}
    points,selection=_read_strict_source_points(Path(inputs['colmap_sparse_points']['path']),{prior.cameras[f].image_id for f in prior.records},maximum_reprojection_error=2.,minimum_track_views=2)
    fp=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene/'exact_render_query_threshold96/field.pt'
    field=torch.load(fp,map_location='cpu',weights_only=False);geo.assert_binding(field['geometry_binding'])
    observed=field['state_dict']['observed'].to(a.device).float();del field
    prediction_bindings={}
    rows=[];views=[]
    for split,frames in [('mapping',report['mapping_frames']),('heldout',report['heldout_frames'])]:
        for frame in frames:
            cam=prior.cameras[frame];w=128;h=round(cam.height*w/cam.width)
            k=cam.intrinsic.to(a.device).clone();k[0]*=w/cam.width;k[1]*=h/cam.height
            pose=cam.world_to_camera.to(a.device);z=(geo.xyz@pose[:3,:3].T+pose[:3,3])[:,2]
            colors=torch.cat([geo.xyz,z[:,None].square(),observed[:,None]],1)
            with torch.no_grad():
                im,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,viewmats=pose[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
                mass=alpha[0,:,:,0];mean=im[0]/mass[:,:,None].clamp_min(1e-8)
            rr=prior.records[frame];assert sha256_file(rr['prediction'])==rr['prediction_sha256'];prediction_bindings[rr['prediction']]=rr['prediction_sha256']
            prediction=torch.load(rr['prediction'],map_location='cpu',weights_only=False);cal=prior.calibration[frame]
            native_depth=prediction['point_map'][...,2].float()*prior.scale*cal['scale']+cal['offset']
            candidates=covered=0
            for pid,xy in zip(cam.point_ids.tolist(),cam.xy):
                if pid not in points:continue
                uv=xy*torch.tensor([w/cam.width,h/cam.height]);x,y=uv.floor().long().tolist()
                if not (0<=x<w and 0<=y<h):continue
                reference=points[pid]@cam.world_to_camera[:3,:3].T+cam.world_to_camera[:3,3]
                if reference[2]<=0:continue
                candidates+=1
                if mass[y,x]<.5:continue
                covered+=1
                world=mean[y,x,:3];camera=world@pose[:3,:3].T+pose[:3,3];depth=camera[2]
                variance=(mean[y,x,3]-depth.square()).clamp_min(0)
                error=float((depth-reference[2].to(depth)).abs()/reference[2].to(depth))
                valid=bool(_bilinear(prediction['validity'].float(),xy[None],cam.width,cam.height)[0]>.999)
                native_value=float(_bilinear(native_depth,xy[None],cam.width,cam.height)[0])
                native_absrel=abs(native_value-float(reference[2]))/float(reference[2]) if valid and cal['accepted'] and native_value>0 else None
                rows.append(dict(native_absrel=native_absrel,semantic_support_fraction=float(mean[y,x,4]),frame=frame,split=split,point_id=pid,absrel=error,depth_cv=float(variance.sqrt()/depth.clamp_min(1e-8)),alpha=float(mass[y,x])))
            views.append(dict(frame=frame,split=split,candidates=candidates,covered=covered))
    groups={}
    for split in ['mapping','heldout']:
        selected=[r for r in rows if r['split']==split];vv=[r for r in views if r['split']==split];total=sum(r['candidates'] for r in vv);covered=sum(r['covered'] for r in vv)
        groups[split]=dict(unique_points=len({r['point_id'] for r in selected}),coverage=covered/total if total else None,candidate_observations=total,absrel=stats([r['absrel'] for r in selected]),depth_cv=stats([r['depth_cv'] for r in selected]),fraction_absrel_over_005=sum(r['absrel']>.05 for r in selected)/len(selected) if selected else None)
    support_groups={}
    for split in ['mapping','heldout']:
        for support in [False,True]:
            selected=[r for r in rows if r['split']==split and (r['semantic_support_fraction']>.5)==support]
            matched=[r for r in selected if r['native_absrel'] is not None]
            support_groups[f'{split}_semantic_{support}']=dict(unique_points=len({r['point_id'] for r in selected}),absrel=stats([r['absrel'] for r in selected]),matched_gaussian=stats([r['absrel'] for r in matched]),matched_moge=stats([r['native_absrel'] for r in matched]))
    result=dict(support_groups=support_groups,prediction_bindings=prediction_bindings,field_sha256=sha256_file(fp),schema='all_source_track_render.v2',scene=a.scene,source_only=True,benchmark_data_used=False,geometry_unchanged=True,raw_alpha_compositor=True,geometry_sha256=g['geometry_sha256'],geometry_report_sha256=sha256_file(gp),implementation_sha256=sha256_file(__file__),source_geometry_receipt_inputs=prior.report['geometry_receipt']['inputs'],selection=selection,groups=groups,views=views,rows=rows,limitations='sparse textured tracks are not dense GT; heldout views excluded from surface fusion but existing calibration may have used their sparse observations; source-centre depth moments do not integrate within Gaussian thickness; no semantic teacher depth gate; repeated observations not independent points')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,groups,'support',support_groups,flush=True)


if __name__=='__main__':run()
