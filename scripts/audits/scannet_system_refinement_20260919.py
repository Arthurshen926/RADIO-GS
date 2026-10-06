"""Fixed native/scope score compatibility controls, all8 same-geometry evaluation."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS,NYU40_ID_TO_NAME
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels,volume_weighted_split_metrics
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
from radio_gs.utils.immutable_artifacts import sha256_file
SCENES=['scene0000_00','scene0062_00','scene0070_00','scene0097_01','scene0140_00','scene0347_00','scene0400_00','scene0590_00']
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('output/scannet_system_refinement_20260919'));p.add_argument('--scenes',nargs='+',default=SCENES);a=p.parse_args();torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False)
registration=dict(controls={'equal_logit':'0.5*native_calibrated_logit+0.5*final_scope_logit','bounded_logit':'native_calibrated_logit+clamp(final_scope_logit-native_calibrated_logit,-1,+1)'},native_calibration='Existing source-fitted thresholds and temperature, not new fitting',selection='Both registered controls evaluated on all8; no scene-specific winners or GT hyperparameters.',source_check='Two declared source-validation camera views; teacher region descriptors define pseudo class preference. Diagnostic teacher self-consistency, no independent GT.',geometry_frozen=True,unobserved_void=True,target_gt_before_predictions=False,implementation_sha256=sha256_file(__file__))
(a.output/'registration.json').write_text(json.dumps(registration,indent=2)+'\n');execution=[]
with torch.no_grad():
 for scene in a.scenes:
  start=time.time();root=Path('output/scannet_frozen_field_20260918')/scene;r=root/'reconstruction/field';out=a.output/scene;out.mkdir();record=dict(scene=scene,status='running');execution.append(record);(a.output/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
  report=json.loads((r/'bounded_fallback_admission/report.json').read_text());fp=r/'bounded_fallback_admission/field.pt';assert sha256_file(fp)==report['field_sha256'];payload=torch.load(fp,map_location='cpu',weights_only=False);model=ElementRetrievalField(**payload['architecture']).cuda();model.load_state_dict(payload['state_dict']);model.requires_grad_(False);bank=torch.load(root/'official_protocol_evaluation/queries.pt',map_location='cpu',weights_only=False);classes=sorted(set(sum([SPLITS[s] for s in ['19','15','10']],[])));q=bank['encoding'][[bank['texts'].index(NYU40_ID_TO_NAME[c]) for c in classes]].float().cuda();cos=model.native_cosine(q);nl=model.readout_log_scale['text'].clamp(-4,6).exp()*(cos-model.native_query_thresholds['text'](q));sl=model.logits(q,'text');scores=dict(native=cos,calibrated_native=nl,scope=sl.sigmoid(),equal_logit=.5*(nl+sl),bounded_logit=nl+(sl-nl).clamp(-1,1));predictions={};choice={}
  for name,score in scores.items():
   predictions[name]={};choice[name]={}
   for split in ['19','15','10']:
    ids=SPLITS[split];ix=[classes.index(c) for c in ids];label=torch.tensor(ids,device='cuda:0')[score[:,ix].argmax(1)];label=torch.where(model.observed,label,0);predictions[name][split]=label.cpu().numpy();choice[name][split]=label
   np.savez_compressed(out/(name+'.npz'),**predictions[name])
  ref=np.load(root/'official_protocol_evaluation/predictions.npz');scope_ref=np.load(root/'official_protocol_evaluation/object_scope_control.npz')
  for s in ['19','15','10']:assert np.array_equal(ref[s],predictions['native'][s]);assert np.array_equal(scope_ref[s],predictions['scope'][s])
  # Source diagnostic is generated before target semantic GT access.
  info=json.loads((r/'teachers.json').read_text());rec=info['teacher_files']['siglip'];assert sha256_file(r/rec['path'])==rec['sha256'];desc=F.normalize(torch.load(r/rec['path'],map_location='cpu',weights_only=False)['descriptors'].float().cuda(),dim=1);assert sha256_file(r/'episodes.pt')==json.loads((r/'episodes.json').read_text())['sha256'];ep=torch.load(r/'episodes.pt',map_location='cpu',weights_only=False);auth=json.loads((r/'responsibility.json').read_text());assert sha256_file(r/'responsibility.json')==info['authority_sha256'];h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width'];rows=[]
  for frame in info['validation_frames']:
   vi=info['frames'].index(frame);ids=torch.where(ep['proposal_views']==vi)[0];rec=auth['views'][vi];assert sha256_file(r/rec['relative_path'])==rec['sha256'];hit=torch.load(r/rec['relative_path'],map_location='cuda:0',weights_only=False);pid,gid,value=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float();keep=model.observed[gid];pid,gid,value=pid[keep],gid[keep],value[keep];mass=torch.zeros(h*w,device='cuda:0').scatter_add_(0,pid,value);op=torch.sparse_coo_tensor(torch.stack([pid,gid]),value/mass[pid].clamp_min(1e-8),(h*w,len(model.observed))).coalesce();mask=ep['masks'][ids].reshape(len(ids),-1).float().cuda();den=mask.sum(0);target_score=F.normalize(mask.T@desc[ids]/den[:,None].clamp_min(1),dim=1)@q.T;valid=(den>0)&(mass>0);metrics={}
   for split in ['19','15','10']:
    cl=SPLITS[split];ix=[classes.index(c) for c in cl];truth=target_score[:,ix].argmax(1);metrics[split]={}
    for name in scores:
     votes=torch.stack([(choice[name][split]==c).float() for c in cl],1);prediction=torch.sparse.mm(op,votes).argmax(1);ious=[]
     for c in range(len(cl)):
      yes=(truth==c)&valid;pp=(prediction==c)&valid
      if yes.any():ious.append(float((yes&pp).sum()/(yes|pp).sum().clamp_min(1)))
     metrics[split][name]=dict(agreement=float((prediction[valid]==truth[valid]).float().mean()),miou=float(np.mean(ious)))
   rows.append(dict(frame=frame,valid_pixels=int(valid.sum()),metrics=metrics))
  source={s:{name:{metric:float(np.mean([row['metrics'][s][name][metric] for row in rows])) for metric in ['agreement','miou']} for name in scores} for s in ['19','15','10']};source_report=dict(source_only=True,teacher_self_consistency_not_gt=True,frames=info['validation_frames'],rows=rows,macro=source);(out/'source_validation.json').write_text(json.dumps(source_report,indent=2)+'\n')
  # Exact fixed official evaluation after all predictions and source checks persisted.
  gr=json.loads((root/'geometry_report.json').read_text())['results']['independent_scale'];geo=FrozenGaussianGeometry.load(gr['geometry'],expected_sha256=gr['geometry_sha256'],device='cpu');geo.assert_binding(payload['geometry_binding']);gtpath=Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/(scene+'_vh_clean_2.labels.ply');xyz,labels=_read_label_ply(gtpath);gt,audit=assign_vala_pseudo_labels(geo.xyz.numpy(),geo.scales.numpy(),geo.quaternions.numpy(),xyz,labels,radius_factor=5.,candidate_k=1000,fallback_k=1,class_balance=True,chunk_size=512);weights=(geo.scales.prod(1)*geo.opacities.flatten()).numpy();metrics={name:{s:volume_weighted_split_metrics(gt,predictions[name][s],weights,SPLITS[s]) for s in ['19','15','10']} for name in scores};old=json.loads((root/'official_protocol_evaluation/report.json').read_text())
  for s in ['19','15','10']:assert abs(metrics['native'][s]['miou']-old['metrics'][s]['miou'])<1e-10
  result=dict(scene=scene,registration_sha256=sha256_file(a.output/'registration.json'),field_sha256=report['field_sha256'],geometry_sha256=gr['geometry_sha256'],gt_sha256=sha256_file(gtpath),query_sha256=sha256_file(root/'official_protocol_evaluation/queries.pt'),baseline_exact=True,predictions_saved_before_gt=True,source_validation_sha256=sha256_file(out/'source_validation.json'),prediction_sha256={name:sha256_file(out/(name+'.npz')) for name in scores},source_macro=source,metrics=metrics,scope_source_acceptance=json.loads((root/'execution.json').read_text()).get('source_acceptance'),promotion=False,independent_test=False,pseudo_gt_audit=audit)
  (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');record.update(status='complete',seconds=time.time()-start,report_sha256=sha256_file(out/'report.json'));(a.output/'execution.json').write_text(json.dumps(execution,indent=2)+'\n');print(scene,{name:{s:round(metrics[name][s]['miou']*100,3) for s in ['19','15','10']} for name in scores},flush=True);del model,payload,scores,desc,ep;torch.cuda.empty_cache()
print('complete',flush=True)
