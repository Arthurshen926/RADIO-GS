"""Held-out point-ID scale audit, using source tracks only and native validity."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points,_bilinear
from radio_gs.v4.geometry.depth_calibration import fit_constrained_affine_depth
from radio_gs.geometry.source_camera_assets import source_colmap_assets
from radio_gs.utils.immutable_artifacts import sha256_file


def stats(x):
 return dict(n=len(x),median=float(x.median()),p90=float(x.quantile(.9)),mean=float(x.mean()),within_5pct=float((x<.05).float().mean()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--geometry-calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
 prior=MogeGeometryPrior(a.geometry_calibration,sha256_file(a.geometry_calibration));sparse,_=source_colmap_assets(Path(prior.manifest['scene_root']))
 pts,pfilter=_read_strict_source_points(sparse/'points3D.bin',{prior.cameras[f].image_id for f in prior.records})
 rows=[];ratios=[]
 for frame,rec in sorted(prior.records.items()):
  assert sha256_file(rec['prediction'])==rec['prediction_sha256'];blob=torch.load(rec['prediction'],weights_only=False);cam=prior.cameras[frame]
  ids=[i for i,pid in enumerate(cam.point_ids.tolist()) if pid in pts]
  pid=cam.point_ids[ids];xy=cam.xy[ids];xyz=torch.stack([pts[int(i)] for i in pid]);ref=(xyz@cam.world_to_camera[:3,:3].T+cam.world_to_camera[:3,3])[:,2]
  pred=_bilinear(blob['point_map'][...,2].float(),xy,cam.width,cam.height,pixel_contract='image_edge_v2')
  mask=_bilinear(blob['validity'].float(),xy,cam.width,cam.height,pixel_contract='image_edge_v2')>.999
  finite=torch.isfinite(pred)&torch.isfinite(ref)&(pred>0)&(ref>0);valid=finite&mask
  train=valid&(pid%5!=0);test=valid&(pid%5==0)
  if train.sum()<32 or test.sum()<16:continue
  ratio=(ref[train]/pred[train]).median();ratios.append(ratio)
  rows.append(dict(frame=frame,pred=pred,ref=ref,train=train,test=test,invalid_fraction=float((finite&~mask).sum()/finite.sum()),ratio=ratio))
 if len(rows)<3:raise ValueError('insufficient independent track-ID split')
 unit=torch.stack(ratios).median();results=[];aggregate={k:[] for k in ['global','scale','raw_affine','normalized_affine']}
 for r in rows:
  x=r['pred']*unit;y=r['ref'];train=r['train'];test=r['test'];local=float(r['ratio']/unit)
  raw=fit_constrained_affine_depth(x[train],y[train]);norm=y[train].median()
  normalized=fit_constrained_affine_depth(x[train]/norm,y[train]/norm)
  candidates=dict(global_prediction=x,scale=x*local,raw_affine=x*raw.scale+raw.offset,normalized_affine=x*normalized.scale+normalized.offset*norm)
  metric={}
  for k,value in candidates.items():
   name='global' if k=='global_prediction' else k;error=(value[test]-y[test]).abs()/y[test];metric[name]=stats(error);aggregate[name].append(error)
  results.append(dict(frame=r['frame'],train_points=int(train.sum()),heldout_points=int(test.sum()),invalid_prediction_fraction=r['invalid_fraction'],pure_scale=local,raw_affine=dict(scale=raw.scale,offset=raw.offset,accepted=raw.accepted),normalized_affine=dict(scale=normalized.scale,offset=float(normalized.offset*norm),accepted=normalized.accepted),heldout=metric))
 output=dict(schema='source_scale_point_holdout.v1',source_only=True,benchmark_data_used=False,split='global point ID modulo 5 == 0 held out from global and local fitting',source_calibration_sha256=sha256_file(a.geometry_calibration),global_train_only_scale=float(unit),prior_global_scale=prior.scale,point_filter=pfilter,rows=results,aggregate={k:stats(torch.cat(v)) for k,v in aggregate.items()},limitations='Sparse source COLMAP reference; poses/SfM reused; no independent dense geometry truth. Normalized affine is a diagnostic, not automatic acceptance.',implementation_sha256=sha256_file(__file__))
 a.output.write_text(json.dumps(output,indent=2)+'\n');print(output['aggregate'],flush=True)
if __name__=='__main__':main()
