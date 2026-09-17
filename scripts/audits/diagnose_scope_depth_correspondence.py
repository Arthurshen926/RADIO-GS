"""Depth-consistent source-group transfer, independent of Gaussian mask lifting."""
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';out=root.parent/'scope_depth_correspondence.json';assert not out.exists();info=json.loads((root/'teachers.json').read_text());ep=root/'caption_region_evidence.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];e=torch.load(ep,weights_only=False);assert e['teacher_manifest_sha256']==sha256_file(root/'teachers.json');gp=Path('output/geometry_only_repair_final_20260914')/a.scene/'report.json';gr=json.loads(gp.read_text());prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);recovered={r['frame']:r for r in gr['recovered_scales'] if r['accepted']};groupspath=root/'grounded_semantic_transfer96/report.json';groups=json.loads(groupspath.read_text())['groups'];H,W=e['masks'].shape[-2:];y,x=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij');pixel=torch.stack([x+.5,y+.5,torch.ones_like(x)],-1).float().reshape(-1,3);views={};inputs={str(ep):sha256_file(ep),str(gp):sha256_file(gp),str(groupspath):sha256_file(groupspath)};unavailable=[]
for frame in info['train_frames']:
 cal=prior.calibration[frame]
 if cal['accepted']:scale=cal['scale'];offset=cal['offset'];kind='accepted_local'
 elif frame in recovered:scale=recovered[frame]['scale'];offset=0.;kind='accepted_neighbor_recovery'
 else:unavailable.append(frame);continue
 rec=prior.records[frame];assert sha256_file(rec['prediction'])==rec['prediction_sha256'];inputs[rec['prediction']]=rec['prediction_sha256'];raw=torch.load(rec['prediction'],weights_only=False);depth=raw['point_map'][...,2].float()*prior.scale*scale+offset;valid=raw['validity'].bool()&torch.isfinite(depth)&(depth>0);depth=F.interpolate(depth[None,None],size=(H,W),mode='nearest-exact')[0,0].flatten();valid=F.interpolate(valid[None,None].float(),size=(H,W),mode='nearest-exact')[0,0].flatten().bool();camera=prior.cameras[frame];k=camera.intrinsic.clone();k[0]*=W/camera.width;k[1]*=H/camera.height;pose=camera.camera_to_world;world=(pixel@torch.linalg.inv(k).T)*depth[:,None];world=world@pose[:3,:3].T+pose[:3,3];vi=info['frames'].index(frame);targets={j:e['masks'][[i for i in g['regions'] if int(e['proposal_views'][i])==vi]].bool().any(0).flatten() for j,g in enumerate(groups) if vi in g['views']};views[frame]=dict(depth=depth,valid=valid,k=k,pose=pose,world=world,targets=targets,calibration_kind=kind)
reference_path=root.parent/'scope_transfer_weighted_stages/report.json';reference=json.loads(reference_path.read_text());inputs[str(reference_path)]=sha256_file(reference_path);ref={(r['frame'],r['selected_group']):r for r in reference['rows'] if r['selected_2d']['iou']==1 and r['query_index']==groups[r['selected_group']]['source_query_index']};rows=[];pairs=[]
for frame,target in views.items():
 ids=sorted(target['targets']);total=torch.zeros(H*W,len(ids));mass=torch.zeros_like(total)
 for sf,source in views.items():
  if sf==frame:continue
  cam=(target['world']-source['pose'][:3,3])@source['pose'][:3,:3];uv=cam@source['k'].T;sx=(uv[:,0]/uv[:,2].clamp_min(1e-8)).floor().long();sy=(uv[:,1]/uv[:,2].clamp_min(1e-8)).floor().long();inside=(cam[:,2]>0)&(sx>=0)&(sx<W)&(sy>=0)&(sy<H);index=sy.clamp(0,H-1)*W+sx.clamp(0,W-1);valid=inside&target['valid']&source['valid'][index];residual=(cam[:,2]-source['depth'][index]).abs();consistent=valid&(residual<.08*prior.scale+.02*source['depth'][index]);pairs.append(dict(target=frame,source=sf,overlap=int(valid.sum()),depth_consistent_fraction=float(consistent.sum()/valid.sum().clamp_min(1))))
  for col,j in enumerate(ids):
   if j not in source['targets']:continue
   total[:,col]+=source['targets'][j][index]*consistent;mass[:,col]+=consistent
 for col,j in enumerate(ids):
  if (frame,j) not in ref or not any(j in v['targets'] for f,v in views.items() if f!=frame):continue
  truth=target['targets'][j];pred=(total[:,col]/mass[:,col].clamp_min(1)>.5)&(mass[:,col]>0);r=ref[frame,j];rows.append(dict(frame=frame,group=j,phrase=groups[j]['phrase'],depth_iou=float((pred&truth).sum()/(pred|truth).sum().clamp_min(1)),depth_positive_support=float((mass[:,col][truth]>0).float().mean()),gaussian_weighted_iou=r['weighted_lvo']['iou'],gaussian_conditional_iou=r['union_lvo']['iou'],gaussian_positive_support=r['lvo_positive_support']))
 print(a.scene,frame,len(rows),flush=True)
macro={k:sum(r[k] for r in rows)/len(rows) for k in ['depth_iou','depth_positive_support','gaussian_weighted_iou','gaussian_conditional_iou','gaussian_positive_support']};result=dict(schema='scope_depth_correspondence.v1',source_only=True,benchmark_data_used=False,diagnostic_only=True,target_source_depth_used=True,target_masks_for_scoring_only=True,independent_geometry_truth=False,inputs=inputs,macro=macro,rows=rows,depth_pairs=pairs,unavailable_calibration_frames=unavailable,calibration_kinds={f:v['calibration_kind'] for f,v in views.items()},implementation_sha256=sha256_file(__file__),limitation='Same observation cohort and target masks; support differs. Direct depth uses target source depth and bypasses fused geometry, so this is not a deployable retrieval baseline or isolated proof of geometry error. Missing groups are unknown; source depth tolerance fixed .08m + .02 depth.')
out.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,macro,flush=True)
