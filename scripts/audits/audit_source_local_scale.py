"""Source COLMAP track transport through frozen Gaussian contributions.

Measure geometric correspondence without masks or semantic features. Include a
snapped-pixel, true-track-depth control to expose 96x128 quantization effects.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.geometry.local_depth_calibration import local_depth_scales
from radio_gs.geometry.source_scale_calibration import fit_source_depth_scale, fit_reprojection_scale
from radio_gs.v4.geometry.fuse_lerf_moge3 import _bilinear


def summary(values):
    x=torch.tensor(values)
    return dict(count=len(x),median=float(x.median()),p90=float(x.quantile(.9)),mean=float(x.mean()),fraction_over_2_pixels=float((x>2).float().mean())) if len(x) else dict(count=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
    base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=base/'preserved_source_inventory_20260916'/a.scene
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    gp=base/'geometry_only_repair_final_20260914'/a.scene/'report.json';gr=json.loads(gp.read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cpu');geo.assert_binding(auth['metadata']['frozen_geometry_binding'])
    inputs={r['role']:r for r in prior.report['geometry_receipt']['inputs']};pointpath=Path(inputs['colmap_sparse_points']['path']);points,stats=_read_strict_source_points(pointpath,{prior.cameras[f].image_id for f in prior.records},maximum_reprojection_error=2.,minimum_track_views=2)
    H,W=auth['metadata']['feature_height'],auth['metadata']['feature_width'];P=H*W;observations={};bindings={};scale_fits=[]
    target_observations={}
    for frame in info['train_frames']:
        cam=prior.cameras[frame]; k=cam.intrinsic.clone(); k[0]*=W/cam.width; k[1]*=H/cam.height
        for pid3,xy in zip(cam.point_ids.tolist(),cam.xy):
            if pid3 in points and pid3%5!=0:
                uv=xy*torch.tensor([W/cam.width,H/cam.height])
                if 0<=uv[0]<W and 0<=uv[1]<H:target_observations.setdefault(pid3,[]).append((frame,cam.world_to_camera,k,uv))
    for frame in info['train_frames']:
        vi=info['frames'].index(frame);cam=prior.cameras[frame];k=cam.intrinsic.clone();k[0]*=W/cam.width;k[1]*=H/cam.height
        rec=auth['views'][vi];path=root/rec['relative_path'];assert sha256_file(path)==rec['sha256'];bindings[str(path)]=rec['sha256'];hit=torch.load(path,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['base_weights'].float()
        pm=torch.zeros(P).scatter_add_(0,pid,weight);centroid=torch.zeros(P,3).index_add_(0,pid,geo.xyz[gid]*weight[:,None])/pm[:,None].clamp_min(1e-8)
        cal=prior.calibration[frame];rr=prior.records[frame];assert sha256_file(rr['prediction'])==rr['prediction_sha256'];bindings[rr['prediction']]=rr['prediction_sha256'];raw=torch.load(rr['prediction'],weights_only=False)
        native=raw['point_map'][...,2].float()*prior.scale
        training=[i for i,pid3 in enumerate(cam.point_ids.tolist()) if pid3 in points and pid3%5!=0]
        sample=_bilinear(native,cam.xy[training],cam.width,cam.height)
        reference=torch.tensor([float(points[int(cam.point_ids[i])]@cam.world_to_camera[2,:3]+cam.world_to_camera[2,3]) for i in training])
        # Respect native depth validity at the same sampled image-edge rays.
        sample_valid=_bilinear(raw['validity'].float(),cam.xy[training],cam.width,cam.height)>.999
        fit=fit_source_depth_scale(sample[sample_valid],reference[sample_valid]);scale_fits.append(dict(frame=frame,previous_calibration_accepted=cal['accepted'],**fit))
        raw_small=F.interpolate(native[None,None],size=(H,W),mode='nearest-exact')[0,0].flatten()
        valid=F.interpolate(raw['validity'][None,None].float(),size=(H,W),mode='nearest-exact')[0,0].flatten().bool()&torch.isfinite(raw_small)&(raw_small>0)
        rays=[]; poses=[]; intrinsics=[]; pixels=[]
        for pid3,xy in zip(cam.point_ids.tolist(),cam.xy):
            if pid3 not in target_observations:continue
            uv=xy*torch.tensor([W/cam.width,H/cam.height]); x,y=uv.floor().long().tolist()
            if not (0<=x<W and 0<=y<H) or not valid[y*W+x]:continue
            ray=torch.tensor([x+.5,y+.5,1.])@torch.linalg.inv(k).T
            for target_frame,pose,kt,uvt in target_observations[pid3]:
                if target_frame==frame:continue
                rays.append((ray*raw_small[y*W+x])@cam.camera_to_world[:3,:3].T)
                poses.append(pose);intrinsics.append(kt);pixels.append(uvt)
        opt=None
        if rays and fit['accepted']:
            opt=fit_reprojection_scale(torch.stack(rays),cam.camera_to_world[:3,3],torch.stack(poses),torch.stack(intrinsics),torch.stack(pixels),initial_scale=fit['scale'])
        scale_fits[-1]['reprojection_fit']=opt
        optimized=raw_small*opt['scale'] if opt and opt['fitted'] else None
        depth=raw_small*cal['scale']+cal['offset'] if cal['accepted'] else None
        revised=raw_small*fit['scale'] if fit['accepted'] else None
        for pid3,xy in zip(cam.point_ids.tolist(),cam.xy):
            if pid3 not in points or pid3%5!=0:continue
            uv=xy*torch.tensor([W/cam.width,H/cam.height]);x,y=uv.floor().long().tolist()
            if not (0<=x<W and 0<=y<H):continue
            pixel=y*W+x
            if pm[pixel]<=1e-8:continue
            world=points[pid3];camera_point=world@cam.world_to_camera[:3,:3].T+cam.world_to_camera[:3,3]
            ray=torch.tensor([x+.5,y+.5,1.])@torch.linalg.inv(k).T
            def backproject(d):return (ray*d)@cam.camera_to_world[:3,:3].T+cam.camera_to_world[:3,3]
            exact_native=_bilinear(native,xy[None],cam.width,cam.height)[0]
            exact_valid=bool(_bilinear(raw['validity'].float(),xy[None],cam.width,cam.height)[0]>.999) and bool(torch.isfinite(exact_native)&(exact_native>0))
            exact_ray=torch.cat([uv,torch.ones(1)])@torch.linalg.inv(k).T
            def exact_backproject(d):return (exact_ray*d)@cam.camera_to_world[:3,:3].T+cam.camera_to_world[:3,3]
            exact_old=exact_backproject(exact_native*cal['scale']+cal['offset']) if exact_valid and cal['accepted'] else None
            exact_optimized=exact_backproject(exact_native*opt['scale']) if exact_valid and opt and opt['fitted'] else None
            local_scale,local_known=local_depth_scales(cam.xy[training][sample_valid]/torch.tensor([cam.width,cam.height]),sample[sample_valid],reference[sample_valid],torch.tensor([[(x+.5)/W,(y+.5)/H]]),raw_small[pixel:pixel+1],fallback_scale=fit['scale'])
            local=backproject(raw_small[pixel]*local_scale[0].float()) if fit['accepted'] and valid[pixel] else None
            observations.setdefault(pid3,[]).append(dict(local=local,local_known=bool(local_known[0]),exact_moge=exact_old,exact_optimized=exact_optimized,optimized=backproject(optimized[pixel]) if optimized is not None and valid[pixel] else None,frame=frame,xy=uv,k=k,pose=cam.world_to_camera,gaussian=centroid[pixel],quantized=backproject(camera_point[2]),moge=backproject(depth[pixel]) if depth is not None and valid[pixel] else None,repaired=backproject(revised[pixel]) if revised is not None and valid[pixel] else None))
    rows=[]
    for pid3,obs in observations.items():
        if len(obs)<2:continue
        for src in obs:
            for dst in obs:
                if src['frame']==dst['frame']:continue
                result=dict(point_id=pid3,source_frame=src['frame'],target_frame=dst['frame'],local_supported=src['local_known'])
                for name in ['gaussian','quantized','moge','repaired','optimized','exact_moge','exact_optimized','local']:
                    if src[name] is None:result[name+'_pixels']=None;continue
                    camera=src[name]@dst['pose'][:3,:3].T+dst['pose'][:3,3]
                    if camera[2]<=0:result[name+'_pixels']=None;continue
                    uv=camera@dst['k'].T;uv=uv[:2]/uv[2];result[name+'_pixels']=float((uv-dst['xy']).norm())
                rows.append(result)
    macro={name:summary([r[name+'_pixels'] for r in rows if r[name+'_pixels'] is not None]) for name in ['gaussian','quantized','moge','repaired','optimized','exact_moge','exact_optimized','local']}
    paired=[r for r in rows if all(r[name+'_pixels'] is not None for name in ['gaussian','quantized','moge','repaired','optimized','exact_moge','exact_optimized','local'])]
    matched={name:summary([r[name+'_pixels'] for r in paired]) for name in ['gaussian','quantized','moge','repaired','optimized','exact_moge','exact_optimized','local']}
    result=dict(schema='source_local_scale_crossfit.v1',local_implementation_sha256=sha256_file('radio_gs/geometry/local_depth_calibration.py'),local_policy=dict(neighbors=16,minimum_neighbors=4,spatial_radius=.1,log_depth_radius=.2,maximum_ratio=1.2),scale_fits=scale_fits,calibration_split='global point_id modulo 5: zero held out from every new scale fit; others training',fit_implementation_sha256=sha256_file('radio_gs/geometry/source_scale_calibration.py'),scene=a.scene,source_only=True,benchmark_data_used=False,semantic_masks_used=False,diagnostic_only=True,track_identity_not_manual_gt=True,
        resolution=[H,W],strict_point_statistics=stats,shared_points=sum(len(v)>1 for v in observations.values()),macro=macro,matched_accepted_depth=matched,rows=rows,
        geometry_sha256=g['geometry_sha256'],authority_sha256=info['authority_sha256'],geometry_report_sha256=sha256_file(gp),source_geometry_receipt_inputs=prior.report['geometry_receipt']['inputs'],input_bindings=bindings,implementation_sha256=sha256_file(__file__),
        limitations='centroid of source pixel Gaussian contributions is a diagnostic, not exact distribution transport; sparse COLMAP tracks favor textured locations; quantized reference uses the original track depth, not deployable inference')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,macro,'matched',matched,flush=True)


if __name__=='__main__':main()
