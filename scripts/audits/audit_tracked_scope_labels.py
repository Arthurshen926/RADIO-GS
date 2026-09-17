"""Check same-name native teacher labels on strict source-only COLMAP tracks.

No MoGe depth, Gaussian correspondence, or benchmark masks enter label matching.
COLMAP identities remain sparse automatic correspondences, not manual truth.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';out=root.parent/'tracked_scope_labels_20260918.json';assert not out.exists();info=json.loads((root/'teachers.json').read_text());ground=root/'caption_grounded_regions.pt';assert sha256_file(ground)==json.loads(ground.with_suffix('.json').read_text())['sha256'];native=torch.load(ground,weights_only=False);ep=root/'caption_region_evidence.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];e=torch.load(ep,weights_only=False);assert e['grounding_sha256']==sha256_file(ground) and torch.equal(native['masks'],e['masks']);groupfile=root/'grounded_semantic_transfer96/report.json';groups=json.loads(groupfile.read_text())['groups'];gp=Path('output/geometry_only_repair_final_20260914')/a.scene/'report.json';gr=json.loads(gp.read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);roles={r['role']:r for r in prior.report['geometry_receipt']['inputs']};pointpath=Path(roles['colmap_sparse_points']['path']);points,stats=_read_strict_source_points(pointpath,{prior.cameras[f].image_id for f in prior.records},maximum_reprojection_error=2.,minimum_track_views=2);observations={};viewstats=[];inputs={str(ground):sha256_file(ground),str(ep):sha256_file(ep),str(groupfile):sha256_file(groupfile),str(gp):sha256_file(gp),str(pointpath):sha256_file(pointpath)}
for frame in info['train_frames']:
 vi=info['frames'].index(frame);cam=prior.cameras[frame];ids=[];xy=[];errors=[]
 for pid,uv in zip(cam.point_ids.tolist(),cam.xy):
  if pid not in points:continue
  pc=points[pid]@cam.world_to_camera[:3,:3].T+cam.world_to_camera[:3,3]
  if pc[2]<=0:continue
  projected=pc@cam.intrinsic.T;error=float((projected[:2]/projected[2]-uv).norm())
  if error>2:continue
  ids.append(pid);xy.append(uv);errors.append(error)
 if not ids:viewstats.append(dict(frame=frame,track_observations=0));continue
 xy=torch.stack(xy);local=[j for j,g in enumerate(groups) if vi in g['views']];cache={};label_counts={'positive':0,'negative':0,'boundary_unknown':0}
 for j in local:
  members=[i for i in groups[j]['regions'] if int(e['proposal_views'][i])==vi];r=native['records'][members[0]];h,w=r['native_shape'];union=torch.zeros(h,w,dtype=torch.bool)
  for ri in members:
   rec=native['records'][ri];assert rec['native_shape']==[h,w]
   if ri not in cache:cache[ri]=torch.from_numpy(np.unpackbits(rec['packed_native'].numpy(),axis=-1)[:,:w].copy()).bool()
   union|=cache[ri]
  scaled=xy*torch.tensor([w/cam.width,h/cam.height]);xx,yy=scaled.floor().long().unbind(1);rx=max(1,math.ceil(2*w/cam.width));ry=max(1,math.ceil(2*h/cam.height));valid=(xx>=rx)&(xx<w-rx)&(yy>=ry)&(yy<h-ry);index=torch.where(valid)[0]
  if not len(index):continue
  dx,dy=torch.meshgrid(torch.arange(-rx,rx+1),torch.arange(-ry,ry+1),indexing='ij');patch=union[yy[index,None]+dy.flatten(),xx[index,None]+dx.flatten()];positive=patch.all(1);negative=~patch.any(1);sign=torch.where(positive,1,torch.where(negative,-1,0))
  label_counts['positive']+=int(positive.sum());label_counts['negative']+=int(negative.sum());label_counts['boundary_unknown']+=int((sign==0).sum())
  for k,s in zip(index.tolist(),sign.tolist()):observations.setdefault((j,ids[k]),[]).append((frame,s))
 viewstats.append(dict(frame=frame,track_observations=len(ids),groups=len(local),maximum_observation_reprojection_error=max(errors),labels=label_counts));print(a.scene,frame,len(ids),flush=True)
rows=[];pergroup=[]
for (j,pid),obs in observations.items():
 known=[(f,s) for f,s in obs if s]
 if len(known)<2:continue
 positive=[f for f,s in known if s==1];negative=[f for f,s in known if s==-1]
 rows.append(dict(group=j,point_id=pid,phrase=groups[j]['phrase'],positive_frames=positive,negative_frames=negative,conflict=bool(positive and negative)))
for j,g in enumerate(groups):
 rr=[r for r in rows if r['group']==j];pos=[r for r in rr if r['positive_frames']];pergroup.append(dict(group=j,phrase=g['phrase'],multi_view_confident_tracks=len(rr),tracks_with_positive=len(pos),conflicting_tracks=sum(r['conflict'] for r in rr),conflict_fraction_of_positive_tracks=sum(r['conflict'] for r in rr)/len(pos) if pos else None))
positive=[r for r in rows if r['positive_frames']];conflict=[r for r in rows if r['conflict']];result=dict(schema='tracked_native_scope_labels.v1',source_only=True,benchmark_data_used=False,moge_depth_used=False,gaussian_correspondence_used=False,native_teacher_masks=True,maximum_point_mean_reprojection_error=2.,maximum_observation_reprojection_error=2.,boundary_margin='ceil(2 original camera pixels in each native mask axis)',strict_point_statistics=stats,multi_view_group_tracks=len(rows),group_tracks_with_positive=len(positive),conflicting_group_tracks=len(conflict),conflict_fraction_of_positive_group_tracks=len(conflict)/len(positive) if positive else None,groups_with_positive_tracks=sum(r['tracks_with_positive']>0 for r in pergroup),total_groups=len(groups),view_statistics=viewstats,per_group=pergroup,positive_group_tracks=positive,inputs=inputs,source_camera_receipt_inputs=prior.report['geometry_receipt']['inputs'],implementation_sha256=sha256_file(__file__),limitations='Sparse textured COLMAP tracks are automatic identities, not independent physical-instance truth. Native masks after coarse-key dedup are used. Group-track units repeat physical points across synonymous queries; background-only tracks are reported separately and never dilute the positive-track conflict rate.')
out.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,{k:result[k] for k in ['multi_view_group_tracks','group_tracks_with_positive','conflicting_group_tracks','conflict_fraction_of_positive_group_tracks','groups_with_positive_tracks','total_groups']},flush=True)
