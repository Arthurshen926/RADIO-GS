"""Source mask transfer with calibrated depth rays, bypassing Gaussian lifting.

This uses source teacher depth, not independent geometry ground truth. Target
source depth defines rays/visibility only; target masks are used for scoring.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
    base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=base/'preserved_source_inventory_20260916'/a.scene
    info=json.loads((root/'teachers.json').read_text());cp=root/'caption_region_evidence.pt';assert sha256_file(cp)==json.loads(cp.with_suffix('.json').read_text())['sha256'];cap=torch.load(cp,weights_only=False);assert cap['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    gp=base/'geometry_only_repair_final_20260914'/a.scene/'report.json';gr=json.loads(gp.read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    h,w=cap['masks'].shape[1:];y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij');pixel=torch.stack([x+.5,y+.5,torch.ones_like(x)],-1).float().reshape(-1,3)
    views={};predictions={};unavailable=[]
    for frame in info['train_frames']:
        cal=prior.calibration[frame]
        if not cal['accepted']:unavailable.append(frame);continue
        rec=prior.records[frame];assert sha256_file(rec['prediction'])==rec['prediction_sha256'];predictions[rec['prediction']]=rec['prediction_sha256'];raw=torch.load(rec['prediction'],weights_only=False)
        d=raw['point_map'][...,2].float()*prior.scale*cal['scale']+cal['offset'];valid=raw['validity'].bool()&torch.isfinite(d)&(d>0)
        d=F.interpolate(d[None,None],size=(h,w),mode='nearest-exact')[0,0].flatten();valid=F.interpolate(valid[None,None].float(),size=(h,w),mode='nearest-exact')[0,0].flatten().bool()
        camera=prior.cameras[frame];k=camera.intrinsic.clone();k[0]*=w/camera.width;k[1]*=h/camera.height;pose=camera.camera_to_world
        point=(pixel@torch.linalg.inv(k).T)*d[:,None];world=point@pose[:3,:3].T+pose[:3,3];targets={};vi=info['frames'].index(frame)
        for ri in torch.where(cap['proposal_views']==vi)[0].tolist():
            r=cap['records'][ri];mask=cap['masks'][ri].flatten().bool()
            if r['confidence']<.7 or mask.sum()<20:continue
            for qi in r['phrase_indices']:targets[qi]=targets.get(qi,torch.zeros_like(mask))|mask
        views[frame]=dict(depth=d,valid=valid,k=k,pose=pose,world=world,targets=targets)
    rows=[];pairs=[]
    for frame,target in views.items():
        qids=sorted(target['targets']);total=torch.zeros(h*w,len(qids));mass=torch.zeros_like(total)
        for sf,source in views.items():
            cam=(target['world']-source['pose'][:3,3])@source['pose'][:3,:3];uv=cam@source['k'].T
            sx=(uv[:,0]/uv[:,2].clamp_min(1e-8)).floor().long();sy=(uv[:,1]/uv[:,2].clamp_min(1e-8)).floor().long()
            inside=(cam[:,2]>0)&(sx>=0)&(sx<w)&(sy>=0)&(sy<h);index=sy.clamp(0,h-1)*w+sx.clamp(0,w-1)
            valid=inside&target['valid']&source['valid'][index]
            residual=(cam[:,2]-source['depth'][index]).abs()
            # Existing geometry prior tolerance, fixed before mask inspection.
            consistent=valid&(residual<.08*prior.scale+.02*source['depth'][index])
            if sf==frame:
                assert torch.equal(index[target['valid']],torch.arange(h*w)[target['valid']])
                continue
            pairs.append(dict(target_frame=frame,source_frame=sf,overlap_pixels=int(valid.sum()),depth_consistent_fraction=float(consistent.sum()/valid.sum().clamp_min(1)),median_abs_depth_error_metres=float(residual[valid].median()/prior.scale) if valid.any() else None))
            for col,qi in enumerate(qids):
                if qi not in source['targets']:continue
                total[:,col]+=source['targets'][qi][index]*consistent;mass[:,col]+=consistent
        for col,qi in enumerate(qids):
            other=sum(qi in v['targets'] for f,v in views.items() if f!=frame)
            if not other:continue
            truth=target['targets'][qi];score=total[:,col]/mass[:,col].clamp_min(1);pred=(score>.5)&(mass[:,col]>0)
            rows.append(dict(frame=frame,phrase=cap['phrases'][qi],other_source_views=other,iou=float((pred&truth).sum()/(pred|truth).sum().clamp_min(1)),positive_support=float((mass[:,col][truth]>0).float().mean())))
        print(a.scene,frame,'done',flush=True)
    oldpath=Path(f'paper/artifacts/{a.scene}_source_mask_transfer_20260916.json');old=json.loads(oldpath.read_text());oldrows={(r['frame'],r['phrase']):r for r in old['rows']}
    macro=dict(direct_depth_iou=sum(r['iou'] for r in rows)/len(rows),positive_support=sum(r['positive_support'] for r in rows)/len(rows),matched_gaussian_mask_iou=sum(oldrows[(r['frame'],r['phrase'])]['cross_view_iou'] for r in rows)/len(rows))
    out=dict(schema='direct_depth_source_mask_transfer.v1',scene=a.scene,source_only=True,benchmark_data_used=False,diagnostic_only=True,target_source_depth_used=True,target_masks_for_scoring_only=True,
        unavailable_calibration_frames=unavailable,observations=len(rows),macro=macro,rows=rows,depth_pairs=pairs,caption_sha256=sha256_file(cp),geometry_report_sha256=sha256_file(gp),prior_report_sha256=gr['source_report_sha256'],prediction_bindings=predictions,
        gaussian_comparison_sha256=sha256_file(oldpath),implementation_sha256=sha256_file(__file__),protocol='inverse warp target calibrated-depth rays into other source images; nearest masks; source-target depth difference < .08m + .02 source depth; only detected phrases; full source mask denominator; different support from Gaussian control, not a deployable gain')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(a.scene,macro,flush=True)


if __name__=='__main__':main()
