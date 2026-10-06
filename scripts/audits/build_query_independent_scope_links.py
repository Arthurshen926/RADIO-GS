"""Match source ranges across views without using phrase equality or benchmark GT.

Links retain endpoints; downstream validation must exclude heldout endpoints.
No link is declared a verified physical instance, and no transitive union is used.
"""
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--diagnostic-region-ids',type=int,nargs='*',default=[]);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists();ep=a.root/'caption_region_evidence.pt';receipt=json.loads(ep.with_suffix('.json').read_text());assert sha256_file(ep)==receipt['sha256'];e=torch.load(ep,weights_only=False);info=json.loads((a.root/'teachers.json').read_text());assert e['source_only'] and not e['benchmark_data_used'];gr=json.loads(a.geometry_report.read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);recovered={r['frame']:r for r in gr['recovered_scales'] if r['accepted']};H,W=e['masks'].shape[-2:];y,x=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij');pixel=torch.stack([x+.5,y+.5,torch.ones_like(x)],-1).float().reshape(-1,3);views={};inputs={str(ep):sha256_file(ep),str(a.geometry_report):sha256_file(a.geometry_report)}
for frame in info['train_frames']:
 cal=prior.calibration[frame]
 if cal['accepted']:scale,offset=cal['scale'],cal['offset']
 elif frame in recovered:scale,offset=recovered[frame]['scale'],0.
 else:continue
 rec=prior.records[frame];assert sha256_file(rec['prediction'])==rec['prediction_sha256'];inputs[rec['prediction']]=rec['prediction_sha256'];raw=torch.load(rec['prediction'],weights_only=False);depth=raw['point_map'][...,2].float()*prior.scale*scale+offset;valid=raw['validity'].bool()&torch.isfinite(depth)&(depth>0);depth=F.interpolate(depth[None,None],size=(H,W),mode='nearest-exact')[0,0].flatten();valid=F.interpolate(valid[None,None].float(),size=(H,W),mode='nearest-exact')[0,0].flatten().bool();camera=prior.cameras[frame];k=camera.intrinsic.clone();k[0]*=W/camera.width;k[1]*=H/camera.height;pose=camera.camera_to_world;world=(pixel@torch.linalg.inv(k).T)*depth[:,None];world=world@pose[:3,:3].T+pose[:3,3]
 ids=[i for i,r in enumerate(e['records']) if r['frame']==frame and r['confidence']>=.7 and int(e['masks'][i].sum())>=20];unique={}
 for i in ids:unique.setdefault(e['masks'][i].numpy().tobytes(),[]).append(i)
 groups=list(unique.values());masks=torch.stack([e['masks'][r[0]].flatten() for r in groups]).float() if groups else torch.zeros(0,H*W);desc=F.normalize(torch.stack([F.normalize(e['native_descriptors'][r].float(),dim=-1).mean(0) for r in groups]),dim=-1) if groups else torch.zeros(0,e['native_descriptors'].shape[1]);views[frame]=dict(depth=depth,valid=valid,k=k,pose=pose,world=world,groups=groups,masks=masks,desc=desc)
def direction(source,target):
 cam=(target['world']-source['pose'][:3,3])@source['pose'][:3,:3];uv=cam@source['k'].T;sx=(uv[:,0]/uv[:,2].clamp_min(1e-8)).floor().long();sy=(uv[:,1]/uv[:,2].clamp_min(1e-8)).floor().long();inside=(cam[:,2]>0)&(sx>=0)&(sx<W)&(sy>=0)&(sy<H);idx=sy.clamp(0,H-1)*W+sx.clamp(0,W-1);visible=inside&target['valid']&source['valid'][idx];visible &= (cam[:,2]-source['depth'][idx]).abs()<.08*prior.scale+.02*source['depth'][idx]
 sm=source['masks'][:,idx]*visible;tm=target['masks']*visible;inter=sm@tm.T;sa=sm.sum(1);ta=tm.sum(1);iou=inter/(sa[:,None]+ta[None]-inter).clamp_min(1);precision=inter/sa[:,None].clamp_min(1);recall=inter/ta[None].clamp_min(1);eligible=(sa[:,None]>=20)&(ta[None]>=20)&(ta[None]>=.2*target['masks'].sum(1)[None]);return iou,precision,recall,eligible
links=[];diagnostics=[];frames=sorted(views)
for fi,f in enumerate(frames):
 for g in frames[fi+1:]:
  left,right=views[f],views[g]
  if not len(left['groups']) or not len(right['groups']):continue
  li,lp,lr,le=direction(left,right);ri,rp,rr,re=direction(right,left);similarity=left['desc']@right['desc'].T;eligible=le&re.T&(li>=.5)&(ri.T>=.5)&(lp>=.65)&(lr>=.65)&(rp.T>=.65)&(rr.T>=.65)&(similarity>=.7);quality=torch.minimum(li,ri.T).masked_fill(~eligible,-1);best_right=quality.argmax(1);best_left=quality.argmax(0)
  for side,own,other,first,second,sim in [('left',left,right,li,ri.T,similarity),('right',right,left,ri,li.T,similarity.T)]:
   for index,region_ids in enumerate(own['groups']):
    if not set(region_ids)&set(a.diagnostic_region_ids):continue
    geometric=torch.minimum(first[index],second[index]);top=int(geometric.argmax());diagnostics.append(dict(frame=f if side=='left' else g,regions=region_ids,other_frame=g if side=='left' else f,best_geometric_regions=other['groups'][top],minimum_bidirectional_iou=float(geometric[top]),native_similarity=float(sim[index,top]),all_admission_conditions_pass=bool(eligible[index,top] if side=='left' else eligible[top,index])))
  for i,j in enumerate(best_right.tolist()):
   if not eligible[i,j] or int(best_left[j])!=i:continue
   links.append(dict(left_frame=f,right_frame=g,left_regions=left['groups'][i],right_regions=right['groups'][j],forward_iou=float(li[i,j]),reverse_iou=float(ri[j,i]),native_similarity=float(similarity[i,j]),relation='mutual_support_candidate',physical_identity_verified=False))
 print('source view',f,'links',len(links),flush=True)
report=dict(schema='query_independent_scope_links.v1',source_only=True,benchmark_data_used=False,query_text_used_for_link_decisions=False,physical_identity_verified=False,transitive_closure_used=False,links=links,diagnostics=diagnostics,diagnostic_regions_used_for_matching=False,inputs=inputs,implementation_sha256=sha256_file(__file__),criteria='Both depth-consistent IoU>=.5; precision/recall>=.65; >=20 co-visible positive pixels and >=20% target coverage each direction; native descriptor cosine>=.7; mutual best pair within each pair of views. Exact coarse duplicate masks treated as one geometric candidate.',validation_policy='Remove every edge incident to heldout views BEFORE using links; endpoint identity stored explicitly.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(len(links))
