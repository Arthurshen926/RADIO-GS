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


def summary(values):
    x=torch.tensor(values)
    return dict(count=len(x),median=float(x.median()),p90=float(x.quantile(.9)),mean=float(x.mean()),fraction_over_2_pixels=float((x>2).float().mean())) if len(x) else dict(count=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
    base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=base/'preserved_source_inventory_20260916'/a.scene
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    gp=base/'geometry_only_repair_final_20260914'/a.scene/'report.json';gr=json.loads(gp.read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cpu');geo.assert_binding(auth['metadata']['frozen_geometry_binding'])
    inputs={r['role']:r for r in prior.report['geometry_receipt']['inputs']};pointpath=Path(inputs['colmap_sparse_points']['path']);points,stats=_read_strict_source_points(pointpath,{prior.cameras[f].image_id for f in prior.records},maximum_reprojection_error=2.,minimum_track_views=2)
    H,W=auth['metadata']['feature_height'],auth['metadata']['feature_width'];P=H*W;observations={};bindings={}
    for frame in info['train_frames']:
        vi=info['frames'].index(frame);cam=prior.cameras[frame];k=cam.intrinsic.clone();k[0]*=W/cam.width;k[1]*=H/cam.height
        rec=auth['views'][vi];path=root/rec['relative_path'];assert sha256_file(path)==rec['sha256'];bindings[str(path)]=rec['sha256'];hit=torch.load(path,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['base_weights'].float()
        pm=torch.zeros(P).scatter_add_(0,pid,weight);centroid=torch.zeros(P,3).index_add_(0,pid,geo.xyz[gid]*weight[:,None])/pm[:,None].clamp_min(1e-8)
        depth=None;cal=prior.calibration[frame]
        if cal['accepted']:
            rr=prior.records[frame];assert sha256_file(rr['prediction'])==rr['prediction_sha256'];bindings[rr['prediction']]=rr['prediction_sha256'];raw=torch.load(rr['prediction'],weights_only=False)
            d=raw['point_map'][...,2].float()*prior.scale*cal['scale']+cal['offset'];v=raw['validity'].bool()&torch.isfinite(d)&(d>0)
            depth=F.interpolate(d[None,None],size=(H,W),mode='nearest-exact')[0,0].flatten();valid=F.interpolate(v[None,None].float(),size=(H,W),mode='nearest-exact')[0,0].flatten().bool()
        for pid3,xy in zip(cam.point_ids.tolist(),cam.xy):
            if pid3 not in points:continue
            uv=xy*torch.tensor([W/cam.width,H/cam.height]);x,y=uv.floor().long().tolist()
            if not (0<=x<W and 0<=y<H):continue
            pixel=y*W+x
            if pm[pixel]<=1e-8:continue
            world=points[pid3];camera_point=world@cam.world_to_camera[:3,:3].T+cam.world_to_camera[:3,3]
            ray=torch.tensor([x+.5,y+.5,1.])@torch.linalg.inv(k).T
            def backproject(d):return (ray*d)@cam.camera_to_world[:3,:3].T+cam.camera_to_world[:3,3]
            observations.setdefault(pid3,[]).append(dict(frame=frame,xy=uv,k=k,pose=cam.world_to_camera,gaussian=centroid[pixel],quantized=backproject(camera_point[2]),moge=backproject(depth[pixel]) if depth is not None and valid[pixel] else None))
    rows=[]
    for pid3,obs in observations.items():
        if len(obs)<2:continue
        for src in obs:
            for dst in obs:
                if src['frame']==dst['frame']:continue
                result=dict(point_id=pid3,source_frame=src['frame'],target_frame=dst['frame'])
                for name in ['gaussian','quantized','moge']:
                    if src[name] is None:result[name+'_pixels']=None;continue
                    camera=src[name]@dst['pose'][:3,:3].T+dst['pose'][:3,3]
                    if camera[2]<=0:result[name+'_pixels']=None;continue
                    uv=camera@dst['k'].T;uv=uv[:2]/uv[2];result[name+'_pixels']=float((uv-dst['xy']).norm())
                rows.append(result)
    macro={name:summary([r[name+'_pixels'] for r in rows if r[name+'_pixels'] is not None]) for name in ['gaussian','quantized','moge']}
    paired=[r for r in rows if all(r[name+'_pixels'] is not None for name in ['gaussian','quantized','moge'])]
    matched={name:summary([r[name+'_pixels'] for r in paired]) for name in ['gaussian','quantized','moge']}
    result=dict(schema='source_track_geometry_transport.v1',scene=a.scene,source_only=True,benchmark_data_used=False,semantic_masks_used=False,diagnostic_only=True,track_identity_not_manual_gt=True,
        resolution=[H,W],strict_point_statistics=stats,shared_points=sum(len(v)>1 for v in observations.values()),macro=macro,matched_accepted_depth=matched,rows=rows,
        geometry_sha256=g['geometry_sha256'],authority_sha256=info['authority_sha256'],geometry_report_sha256=sha256_file(gp),source_geometry_receipt_inputs=prior.report['geometry_receipt']['inputs'],input_bindings=bindings,implementation_sha256=sha256_file(__file__),
        limitations='centroid of source pixel Gaussian contributions is a diagnostic, not exact distribution transport; sparse COLMAP tracks favor textured locations; quantized reference uses the original track depth, not deployable inference')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,macro,'matched',matched,flush=True)


if __name__=='__main__':main()
