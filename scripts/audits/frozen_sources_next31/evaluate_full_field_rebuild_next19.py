"""Explicit-root full rebuilt field evaluation; unchanged LERF output contracts.

Reference query results are development category unions, NOT official four-query
benchmark scores. Optional graph is rebuilt from the current field/observed rows.
"""
import argparse,json,os,subprocess,sys,shutil
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from radio_gs.utils.immutable_artifacts import sha256_file
SCENES=['figurines','ramen','teatime','waldo_kitchen']
COUNTS=dict(figurines=56,ramen=71,teatime=59,waldo_kitchen=22)

def native_compact_calibrated(field, query):
 """Frozen native decoded direction, with current source-fitted text calibration."""
 q=torch.nn.functional.normalize(query,dim=1)
 threshold=field.native_query_thresholds['text'](q)
 scale=field.readout_log_scale['text'].clamp(-4,6).exp()
 rows=[]
 for z in field.latent[:,:field.native_decoder.in_features].split(4096):
  decoded=field.native_decoder(z)
  cosine=torch.nn.functional.normalize(decoded,dim=1)@q.T
  rows.append(torch.sigmoid(scale*(cosine-threshold)).cpu())
 return torch.cat(rows)*field.observed.cpu()[:,None]

def aggregate(base):
 rows={s:json.loads((base/s/'evaluation/report.json').read_text()) for s in SCENES}
 assert all(r['complete'] for r in rows.values());assert sum(r['observations'] for r in rows.values())==208
 arms={'native_compact_calibrated','final','structure32_graph'}
 for scene,row in rows.items():
  assert set(row['lerf2d'])==arms and set(row['selected'])==arms,(scene,'incomplete diagnostic arms')
  for arm in arms:
   assert len(row['lerf2d'][arm]['observations'])==COUNTS[scene]
   assert row['selected'][arm]['count']==COUNTS[scene] and row['selected'][arm]['missing']==0
 result=dict(complete=True,scenes=SCENES,observations=208,independent_test=False,official_four_query=False,macro={name:dict(lerf2d_iou=float(np.mean([r['lerf2d'][name]['miou'] for r in rows.values()])),lerf2d_ap=float(np.mean([r['lerf2d'][name]['macro_ap'] for r in rows.values()])),lerf2d_localization=float(np.mean([r['lerf2d'][name]['localization_accuracy'] for r in rows.values()])),selected_iou=float(np.mean([r['selected'][name]['miou'] for r in rows.values()])),selected_acc25=float(np.mean([r['selected'][name]['acc025'] for r in rows.values()])),selected_acc50=float(np.mean([r['selected'][name]['acc05'] for r in rows.values()]))) for name in arms},reference_development={s:r['reference_development'] for s,r in rows.items()},reports={s:dict(path=str(base/s/'evaluation/report.json'),sha256=sha256_file(base/s/'evaluation/report.json')) for s in SCENES})
 result['reconstruction_lineage']={s:dict(episode_populations=json.loads((base/s/'lineage_audit.json').read_text())['episode_populations'],lineage_sha256=sha256_file(base/s/'lineage_audit.json'),old_train=rows[s]['old_train'],new_train=rows[s]['new_train'],original_two_validation=rows[s]['new_validation'],old_scope_last4=rows[s]['old_scope_validation'],new_scope_last4=rows[s]['new_scope_validation']) for s in SCENES}
 out=base/'evaluation_summary.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['macro'],indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',choices=SCENES);p.add_argument('--base-root',type=Path,default=Path('output/full_field_rebuild_next19'));p.add_argument('--device',default='cuda:0');p.add_argument('--with-structure-graph',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--summarize',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();torch.set_num_threads(4)
 if a.summarize:return aggregate(a.base_root)
 if not a.scene:p.error('--scene required except --summarize')
 root=a.base_root/a.scene;fieldroot=root/'reconstruction/field';gp=Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json';run='bounded_fallback_admission';out=root/'evaluation'
 plan=dict(scene=a.scene,field=str(fieldroot/run/'field.pt'),geometry_report=str(gp),output=str(out),expected_observations=COUNTS[a.scene],with_structure_graph=a.with_structure_graph,native_compact_calibrated_diagnostic=True,reference_query_contract='GT-masked reference crop plus registered positive/negative reference points; category unions; development only, not official four-query',native_grid=True,selected_pixel_threshold=10,element_threshold=.5,source_budget_changed=True,implementation_sha256=sha256_file(__file__))
 if a.prepare_only:print(json.dumps(plan,indent=2));return
 out.mkdir(parents=True,exist_ok=a.resume)
 if (out/'report.json').exists():raise FileExistsError(out/'report.json')
 if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
 else:(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 fr=json.loads((fieldroot/run/'report.json').read_text());fp=fieldroot/run/'field.pt';assert sha256_file(fp)==fr['field_sha256']
 execution=[]
 if (out/'execution.json').exists():execution=json.loads((out/'execution.json').read_text())
 for task in ['text','reference']:
  for mode in ['online','replay']:
   dst=out/(task+'_'+mode+'.json');cmd=[sys.executable,'scripts/audits/audit_current_query_routes.py','--task',task,'--root',str(fieldroot),'--scene',a.scene,'--geometry-report',str(gp),'--runs',run,'--device',a.device,'--output',str(dst)]
   if mode=='replay':cmd+=['--encoding-replay',str(out/(task+'_online.encodings.pt'))]
   previous=next((r for r in execution if r['task']==task and r['mode']==mode),None)
   if previous is not None:
    assert previous['status']=='complete' and previous['command']==cmd,'Inspect failed functional stage; no automatic overwrite'
    for f,h in previous['outputs'].items():assert sha256_file(f)==h
    continue
   original=root/(task+'_'+mode+'.json');build_receipt=root/'execution.json'
   if original.exists() and build_receipt.exists():
    build=json.loads(build_receipt.read_text());assert build['complete'];stage=next(s for s in build['stages'] if s['name']==task+'_'+mode);assert stage['status']=='complete' and stage['outputs'][str(original)]==sha256_file(original);expected=[v.replace(str(out)+'/',str(root)+'/') if isinstance(v,str) else v for v in cmd];assert stage['command']==expected,(stage['command'],expected)
    final_stage=next(s for s in build['stages'] if s['name']=='final_admission');assert final_stage['outputs'][str(fieldroot/run/'report.json')]==sha256_file(fieldroot/run/'report.json')
    copied={}
    for suffix in ['.json','.routes.json','.encodings.pt']:
     src=original if suffix=='.json' else original.with_suffix(suffix);target=dst if suffix=='.json' else dst.with_suffix(suffix);shutil.copy2(src,target);copied[str(target)]=sha256_file(target)
    proto=original.with_suffix('.protocol.json')
    if proto.exists():shutil.copy2(proto,dst.with_suffix('.protocol.json'))
    execution.append(dict(task=task,mode=mode,command=cmd,status='complete',outputs=copied,reused_completed_new_root=True,source_execution_sha256=sha256_file(build_receipt)));(out/'execution.json').write_text(json.dumps(execution,indent=2)+'\n');continue
   row=dict(task=task,mode=mode,command=cmd,status='running');execution.append(row);(out/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
   with (out/(task+'_'+mode+'.log')).open('x') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
   row.update(returncode=result.returncode,status='complete' if result.returncode==0 else 'failed')
   if result.returncode==0:row['outputs']={str(f):sha256_file(f) for f in [dst,dst.with_suffix('.routes.json'),dst.with_suffix('.encodings.pt')]}
   (out/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
   if result.returncode:raise RuntimeError(str(dst)+' failed; inspect preserved log')
  online=json.loads((out/(task+'_online.json')).read_text());replay=json.loads((out/(task+'_replay.json')).read_text());assert online['runs']==replay['runs']
 # Render stage rejects training tensors and target RGB, while allowing labels only for frozen scoring.
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   path=str(args[0]);forbidden=('teacher_radio.pt','teacher_dino.pt','teacher_siglip.pt','teacher_sam.pt','episodes.pt','caption_region_evidence.pt','grounded_text_supervision.pt')
   if any(x in path for x in forbidden):raise RuntimeError('Training tensors forbidden during map readout')
   if '/lerf_ovs/label/' in path and path.lower().endswith(('.jpg','.png')):raise RuntimeError('Target RGB forbidden')
 sys.addaudithook(guard)
 from gsplat import rasterization
 from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
 from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
 from radio_gs.models.element_retrieval_field import load_element_retrieval_field
 from radio_gs.utils.query_input_identity import query_call_sha256
 from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects,_mask_iou,_localization_hit
 from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES,evaluate_scene
 gr=json.loads(gp.read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);identity=json.loads((fieldroot/'native_query_encoder.json').read_text());auth=json.loads((fieldroot/'responsibility.json').read_text());assert auth['metadata']['frozen_geometry_report_sha256']==sha256_file(gp);field,payload=load_element_retrieval_field(fp,expected_sha256=fr['field_sha256'],geometry=geo,expected_query_encoder=identity,device=a.device)
 bankpath=out/'text_online.encodings.pt';receipt=json.loads((out/'text_online.routes.json').read_text());assert receipt['encoding_bank_sha256']==sha256_file(bankpath);bank=torch.load(bankpath,map_location='cpu',weights_only=False);assert len(bank)==1 and bank[0]['kind']=='text';texts=bank[0]['texts'];assert bank[0]['query_input_sha256']==query_call_sha256('text',(texts,),{},identity)
 with torch.no_grad():
  q=bank[0]['encoding'].half().float().to(a.device);scores={'final':field.retrieve(q,'text')['scores'].cpu(),'native_compact_calibrated':native_compact_calibrated(field,q)};graphmeta=None
  if a.with_structure_graph:
   from scripts.audits.graph_transfer_20260919_structure_runtime import refine_structure_scores
   assert field.latent.shape[1]==96 and field.native_semantic_dim==64
   scores['structure32_graph'],graphmeta=refine_structure_scores(geo.xyz,field.latent[:,64:].detach().cpu(),field.observed.cpu(),scores['final'])
  # Scores, support and selections come only from this newly loaded field and rebuilt graph.
  pack=dict(scores=scores,texts=texts,observed=field.observed.cpu(),geometry_binding=payload['geometry_binding'],field_sha256=fr['field_sha256']);torch.save(pack,out/'scores.pt');torch.save({k:(v>.5)&pack['observed'][:,None] for k,v in scores.items()},out/'selections.pt')
  labels=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/a.scene;frames=load_lerf_objects(labels,frames=SCENE_GT_FRAMES[a.scene]);info=json.loads((fieldroot/'teachers.json').read_text());assert {int(f.split('_')[-1]) for f in frames}.isdisjoint(info['frames']);assert set(texts)=={o.query for objects in frames.values() for o in objects};assert sum(map(len,frames.values()))==COUNTS[a.scene]
  observations={name:[] for name in scores};selected_metrics={}
  for name,values in scores.items():
   two=out/name/'lerf2d';sel=out/name/'selected';two.mkdir(parents=True,exist_ok=a.resume);sel.mkdir(parents=True,exist_ok=a.resume);values=values.to(a.device)
   for frame,objects in frames.items():
    h,w=objects[0].mask.shape;cam=prior.cameras[int(frame.split('_')[-1])];k=cam.intrinsic.to(a.device).clone();k[0]*=w/cam.width;k[1]*=h/cam.height;kwargs=dict(viewmats=cam.world_to_camera.to(a.device)[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
    for obj in objects:
     qi=texts.index(obj.query);colors=torch.stack([values[:,qi],field.observed.float()],1);render,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,**kwargs);mass=render[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);prob=torch.where(valid,render[0,...,0]/mass.clamp_min(1e-8),0).cpu();binary=prob>.5;path=two/(frame+'_'+obj.query+'.png');Image.fromarray(binary.numpy().astype(np.uint8)*255).save(path);observations[name].append(dict(frame=frame,query=obj.query,iou=_mask_iou(torch.from_numpy(obj.mask),binary),ap=float(average_precision_score(obj.mask.flatten(),prob.numpy().flatten())),positive_support_recall=float(valid.cpu().numpy()[obj.mask].mean()),alpha_nonzero_recall=float((alpha[0,...,0]>1e-8).cpu().numpy()[obj.mask].mean()),selected_elements=int(((values[:,qi]>.5)&field.observed).sum()),localization_hit=_localization_hit(torch.nonzero(prob==prob.max()),obj.bboxes),prediction_sha256=sha256_file(path)))
     ids=(values[:,qi]>.5)&field.observed
     if ids.any():
      _,a2,_=rasterization(means=geo.xyz[ids],quats=geo.quaternions[ids],scales=geo.scales[ids],opacities=geo.opacities.flatten()[ids],colors=torch.ones(int(ids.sum()),1,device=a.device),**kwargs);byte=(a2[0,...,0]*255+.5).floor().clamp(0,255).byte().cpu().numpy()
     else:byte=np.zeros((h,w),dtype=np.uint8)
     Image.fromarray(byte).save(sel/(frame+'_'+obj.query+'.png'))
    print(a.scene,name,frame,flush=True)
   gt=Path('output/protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/eval_gt')/a.scene/'gt';selected_metrics[name]=asdict(evaluate_scene(gt,sel,a.scene,threshold=10));assert selected_metrics[name]['missing']==0 and selected_metrics[name]['count']==COUNTS[a.scene]
  geo.assert_binding(auth['metadata']['frozen_geometry_binding'])
 oldinfo=json.loads((Path('output/depth_opacity_pipeline_20260918')/a.scene/'reconstruction/field/teachers.json').read_text());reference=json.loads((out/'reference_online.json').read_text())['runs'][run];result=dict(plan,complete=True,independent_test=False,official_four_query=False,field_sha256=fr['field_sha256'],geometry_sha256=g['geometry_sha256'],encoder_sha256=sha256_file(fieldroot/'native_query_encoder.json'),encoding_bank_sha256=sha256_file(bankpath),scores_sha256=sha256_file(out/'scores.pt'),selection_sha256=sha256_file(out/'selections.pt'),observations=COUNTS[a.scene],cold_functional_replay_exact=True,reference_development=reference,graph=graphmeta,old_train=oldinfo['train_frames'],new_train=info['train_frames'],old_validation=oldinfo['validation_frames'],new_validation=info['validation_frames'],old_scope_validation=oldinfo['train_frames'][-4:],new_scope_validation=info['train_frames'][-4:],lerf2d={n:dict(miou=float(np.mean([r['iou'] for r in rows])),macro_ap=float(np.mean([r['ap'] for r in rows])),mean_positive_support_recall=float(np.mean([r['positive_support_recall'] for r in rows])),mean_alpha_nonzero_recall=float(np.mean([r['alpha_nonzero_recall'] for r in rows])),localization_accuracy=float(np.mean([r['localization_hit'] for r in rows])),observations=rows) for n,rows in observations.items()},selected=selected_metrics,method='Expanded-source complete field reconstruction; optional unchanged structure32 graph, rebuilt on current observed rows',source_acceptance=json.loads((root/'execution.json').read_text()).get('source_acceptance'),metric_sources={p:sha256_file(p) for p in ['radio_gs/scripts/eval_prerendered_lerf_features.py','radio_gs/scripts/eval_opengaussian_lerf_baseline.py','scripts/audits/graph_transfer_20260919_structure_runtime.py']})
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,{n:r['miou'] for n,r in result['lerf2d'].items()},{n:r['miou'] for n,r in selected_metrics.items()})
if __name__=='__main__':main()
