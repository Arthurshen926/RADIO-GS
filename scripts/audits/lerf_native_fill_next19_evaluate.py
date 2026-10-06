"""Evaluate measured native fill with separate observed denominators; never tune."""
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES,evaluate_scene
from radio_gs.utils.immutable_artifacts import sha256_file

torch.set_num_threads(4);out=Path('output/lerf_native_fill_next19')
assert json.loads((out/'preparation.json').read_text())['complete']
source_rows=[]
with torch.no_grad():
 # All source validation is completed before opening target annotations.
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  dst=out/scene;root=Path('output/depth_opacity_pipeline_20260918')/scene/'reconstruction/field';info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];ep=torch.load(root/'episodes.pt',map_location='cpu',weights_only=False);p=torch.load(root/'bounded_fallback_admission/field.pt',map_location='cpu',weights_only=False);sd=p['state_dict'];old=sd['latent'][:,:64].cuda();obs=sd['observed'].cuda();fill=torch.load(dst/'native_fill.pt',map_location='cpu',weights_only=False);new=old.clone();ids=fill['added_ids'].cuda();new[ids]=fill['native64'].cuda();valid=obs.clone();valid[ids]=True;rec=info['teacher_files']['siglip'];tp=root/rec['path'];assert sha256_file(tp)==rec['sha256'];desc=F.normalize(torch.load(tp,map_location='cpu',weights_only=False)['descriptors'].float().cuda(),dim=1)@sd['native_decoder.weight'].cuda();h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width'];rows=[]
  for frame in info['validation_frames']:
   vi=info['frames'].index(frame);ix=torch.where(ep['proposal_views']==vi)[0];rec=auth['views'][vi];hp=root/rec['relative_path'];assert sha256_file(hp)==rec['sha256'];hit=torch.load(hp,map_location='cuda:0',weights_only=False);pid,gid,value=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float();op=torch.sparse_coo_tensor(torch.stack([pid,gid]),value,(h*w,len(old))).coalesce();full=torch.sparse.sum(op,dim=1).to_dense();before=torch.sparse.mm(op,old*obs[:,None]);after=torch.sparse.mm(op,new*valid[:,None]);bm=torch.sparse.mm(op,obs.float()[:,None]).flatten();am=torch.sparse.mm(op,valid.float()[:,None]).flatten();masks=ep['masks'][ix].reshape(len(ix),-1).float().cuda();count=masks.sum(0);target=masks.T@desc[ix]/count[:,None].clamp_min(1);seen=(count>0)&(full>1e-8);common=seen&(bm/full.clamp_min(1e-8)>.5);added=seen&(am/full.clamp_min(1e-8)>.5)&~common;bc=F.cosine_similarity(before,target,dim=1);ac=F.cosine_similarity(after,target,dim=1);rows.append({'frame':frame,'common_pixels':int(common.sum()),'added_pixels':int(added.sum()),'common_cosine_before':float(bc[common].mean()),'common_cosine_after':float(ac[common].mean()),'new_pixels_cosine':float(ac[added].mean()) if added.any() else None})
  report={'source_only':True,'teacher_self_consistency_not_gt':True,'validation_frames_not_used_for_fill':True,'rows':rows};(dst/'source_validation.json').write_text(json.dumps(report,indent=2)+'\n');source_rows.append(report);print('source',scene,rows,flush=True)
 # Target masks used only after every prediction and source check has persisted.
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  dst=out/scene;data=torch.load(dst/'scores.pt',map_location='cpu',weights_only=False);prep=json.loads((dst/'preparation.json').read_text());assert sha256_file(dst/'scores.pt')==prep['scores_sha256'];gp=Path('output/depth_opacity_repair_20260918')/scene/'geometry_report.json';gr=json.loads(gp.read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda:0');geo.assert_binding(data['geometry_binding']);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);names=['baseline','measured_fill'];observed=torch.stack([data['observed'][k] for k in names],1).float().cuda();frames=load_lerf_objects(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/scene,frames=SCENE_GT_FRAMES[scene]);rows=[];oldreport=json.loads((Path('output/graph_transfer_final_20260919')/scene/'report.json').read_text());oldrows={(r['frame'],r['query']):r for r in oldreport['observations']};selections={k:(data['scores'][k]>.5)&data['observed'][k][:,None] for k in names};torch.save(selections,dst/'selections.pt')
  for name in names:(dst/name).mkdir(exist_ok=False)
  for frame,objects in frames.items():
   c=prior.cameras[int(frame.split('_')[-1])];h,w=objects[0].mask.shape;k=c.intrinsic.cuda().clone();k[0]*=w/c.width;k[1]*=h/c.height;common={'viewmats':c.world_to_camera.cuda()[None],'Ks':k[None],'width':w,'height':h,'near_plane':.001*prior.scale,'far_plane':100*prior.scale,'packed':False}
   for obj in objects:
    qi=data['texts'].index(obj.query);score=torch.stack([data['scores'][name][:,qi] for name in names],1).cuda()*observed;colors=torch.cat([score,observed],1);render,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,**common);mass=render[0,...,2:];valid=(mass/alpha[0].clamp_min(1e-8)>.5)&(alpha[0]>1e-8);response=torch.where(valid,render[0,...,:2]/mass.clamp_min(1e-8),0).cpu().numpy();metrics={}
    for j,name in enumerate(names):
     pred=response[...,j]>.5;metrics[name]={'iou':float((pred&obj.mask).sum()/max((pred|obj.mask).sum(),1)),'ap':float(average_precision_score(obj.mask.flatten(),response[...,j].flatten())),'support_recall':float(valid[...,j].cpu().numpy()[obj.mask].mean())};sel=selections[name][:,qi].cuda()
     if sel.any():
      _,al,_=rasterization(means=geo.xyz[sel],quats=geo.quaternions[sel],scales=geo.scales[sel],opacities=geo.opacities.flatten()[sel],colors=torch.ones(int(sel.sum()),1,device='cuda'),**common);byte=(al[0,...,0]*255+.5).floor().clamp(0,255).byte().cpu().numpy()
     else:byte=np.zeros((h,w),np.uint8)
     Image.fromarray(byte).save(dst/name/(frame+'_'+obj.query+'.png'))
    reference=oldrows[(frame,obj.query)]['membership_iou']['distilled_structure32_calibrated'];assert abs(reference-metrics['baseline']['iou'])<1e-10,(scene,frame,obj.query,reference,metrics['baseline']);rows.append({'frame':frame,'query':obj.query,'metrics':metrics})
   print('eval',scene,frame,flush=True)
  gt=Path('output/protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/eval_gt')/scene/'gt';selected={name:asdict(evaluate_scene(gt,dst/name,scene,threshold=10)) for name in names};oldsel=json.loads((Path('output/graph_transfer_final_20260919')/scene/'selected/report.json').read_text());baseline_selected=oldsel['results']['distilled_structure32_calibrated'];assert all(selected['baseline'][key]==baseline_selected[key] for key in ['miou','acc025','acc05','count','missing']);assert all(a['iou']==b['iou'] for a,b in zip(selected['baseline']['objects'],baseline_selected['objects']))
  result={'scene':scene,'observations':rows,'count':len(rows),'macro':{name:{m:float(np.mean([r['metrics'][name][m] for r in rows])) for m in ['iou','ap','support_recall']} for name in names},'selected':selected,'source_validation_sha256':sha256_file(dst/'source_validation.json'),'scores_sha256':prep['scores_sha256'],'baseline_exact':True,'geometry_fixed':True,'independent_test':False,'volumetric_3d_iou':False,'implementation_sha256':sha256_file(__file__)};(dst/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',scene,result['macro'],flush=True)
