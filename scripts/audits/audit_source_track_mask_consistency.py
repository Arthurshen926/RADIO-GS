"""Check native teacher masks on shared source-only COLMAP track observations.

Bypasses MoGe depths, Gaussian responsibilities, lifting and feature fields.
Track identity is independent geometric evidence, not verified semantic GT.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
    base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=base/'preserved_source_inventory_20260916'/a.scene
    info=json.loads((root/'teachers.json').read_text());cp=root/'caption_grounded_regions.pt';assert sha256_file(cp)==json.loads(cp.with_suffix('.json').read_text())['sha256'];cap=torch.load(cp,weights_only=False);assert cap['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    gp=base/'geometry_only_repair_final_20260914'/a.scene/'report.json';gr=json.loads(gp.read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    inputs={r['role']:r for r in prior.report['geometry_receipt']['inputs']};pointpath=Path(inputs['colmap_sparse_points']['path']);assert sha256_file(pointpath)==inputs['colmap_sparse_points']['sha256']
    allowed={prior.cameras[f].image_id for f in prior.records}
    points,pointstats=_read_strict_source_points(pointpath,allowed,maximum_reprojection_error=2.,minimum_track_views=2)
    ids=sorted(points);index={p:i for i,p in enumerate(ids)};Q=len(cap['phrases']);N=len(ids);V=len(info['train_frames'])
    labels=torch.full((V,N,Q),-1,dtype=torch.int8);pixel_xy={};viewstats=[]
    for vi,frame in enumerate(info['train_frames']):
        cam=prior.cameras[frame];rows=[i for i,pid in enumerate(cam.point_ids.tolist()) if pid in index]
        pids=cam.point_ids[rows];slot=torch.tensor([index[int(p)] for p in pids]);xy=cam.xy[rows];targets={}
        for ri,r in enumerate(cap['records']):
            if r['frame']!=frame or r['confidence']<.7 or cap['masks'][ri].sum()<20:continue
            native=torch.from_numpy(np.unpackbits(r['packed_native'].numpy(),axis=-1)[:,:r['native_shape'][1]].astype(bool))
            for qi in r['phrase_indices']:targets[qi]=targets.get(qi,torch.zeros_like(native))|native
        if targets:
            H,W=next(iter(targets.values())).shape;x=(xy[:,0]*W/cam.width).floor().long();y=(xy[:,1]*H/cam.height).floor().long();inside=(x>=3)&(x<W-3)&(y>=3)&(y<H-3)
            for qi,target in targets.items():
                t=target.float()[None,None];dilate=F.max_pool2d(t,7,1,3)[0,0].bool();erode=~F.max_pool2d(1-t,7,1,3)[0,0].bool()
                sx=x.clamp(0,W-1);sy=y.clamp(0,H-1);positive=inside&erode[sy,sx];negative=inside&~dilate[sy,sx]
                labels[vi,slot[positive],qi]=1;labels[vi,slot[negative],qi]=0
            pixel_xy[frame]={int(pid):[float(xy[j,0]*W/cam.width),float(xy[j,1]*H/cam.height)] for j,pid in enumerate(pids)}
        viewstats.append(dict(frame=frame,strict_source_tracks=len(rows),grounded_phrases=len(targets)))
        print(a.scene,frame,len(rows),'tracks',flush=True)
    pos=(labels==1).sum(0);neg=(labels==0).sum(0);shared=(pos+neg)>=2;conflict=(pos>0)&(neg>0);positive_shared=shared&(pos>0)
    rows=[];examples=[]
    for qi,phrase in enumerate(cap['phrases']):
        sharedcount=int(shared[:,qi].sum());positivecount=int(positive_shared[:,qi].sum());bad=torch.where(conflict[:,qi])[0]
        if not sharedcount:continue
        rows.append(dict(phrase=phrase,shared_track_concept_pairs=sharedcount,positive_shared_pairs=positivecount,conflicting_pairs=len(bad),conflict_fraction_of_positive_shared=len(bad)/positivecount if positivecount else None))
        for pi in bad[:3].tolist():
            pv=torch.where(labels[:,pi,qi]==1)[0].tolist();nv=torch.where(labels[:,pi,qi]==0)[0].tolist();pid=ids[pi]
            examples.append(dict(phrase=phrase,point_id=pid,positive=[dict(frame=info['train_frames'][v],xy=pixel_xy[info['train_frames'][v]][pid]) for v in pv],negative=[dict(frame=info['train_frames'][v],xy=pixel_xy[info['train_frames'][v]][pid]) for v in nv]))
    result=dict(schema='source_track_mask_consistency.v1',scene=a.scene,source_only=True,benchmark_data_used=False,teacher_masks_are_not_gt=True,
        no_moge_depth_or_gaussian_or_feature_in_computation=True,colmap_track_identity_not_manual_ground_truth=True,maximum_track_reprojection_error=2.,native_boundary_exclusion_radius=3,
        input_bindings={str(cp):sha256_file(cp),str(gp):sha256_file(gp),str(pointpath):sha256_file(pointpath)},source_geometry_receipt_inputs=prior.report['geometry_receipt']['inputs'],
        strict_point_statistics=pointstats,source_views=viewstats,rows=rows,examples=examples,macro=dict(shared_track_concept_pairs=int(shared.sum()),positive_shared_pairs=int(positive_shared.sum()),conflicting_pairs=int(conflict.sum()),conflict_fraction_of_positive_shared=float(conflict.sum()/positive_shared.sum().clamp_min(1))),implementation_sha256=sha256_file(__file__))
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,result['macro'],flush=True)


if __name__=='__main__':main()
