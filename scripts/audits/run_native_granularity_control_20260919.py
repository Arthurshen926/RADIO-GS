"""Fixed source-mask-area strata versus equal-capacity single native field.

Not physical whole/part labels. No trainable query map or GT-driven rule search.
All predictions are materialized before semantic labels are read.
"""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS,NYU40_ID_TO_NAME
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels,volume_weighted_split_metrics
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
from radio_gs.utils.immutable_artifacts import sha256_file

SCENES=['scene0000_00','scene0062_00','scene0070_00','scene0347_00','scene0097_00','scene0400_00','scene0140_00','scene0590_00']
OUT=Path('output/native_granularity_control_20260919');OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
rule=dict(mask_area_fraction_boundaries=[.01,.1],strata=3,dimensions_per_stratum=64,equal_capacity_single_dimension=192,pooling='maximum native cosine over supported strata, then class argmax',source_overlap_weights_unchanged=True,physical_whole_part_labels=False,parameters_fitted_to_gt=False,rule_search=False,independent_test=False,mainline_changed=False,implementation_sha256=sha256_file(__file__))
(OUT/'registration.json').write_text(json.dumps(rule,indent=2)+'\n');record=dict(pid=os.getpid(),cpu_only=True,complete=False,scenes={},failures=[])
def save():
 t=OUT/'execution.tmp';t.write_text(json.dumps(record,indent=2)+'\n');t.replace(OUT/'execution.json')
def load(p,h=None):
 if h:assert sha256_file(p)==h
 return torch.load(p,map_location='cpu',weights_only=False)
def run(scene):
 root=Path('output/scannet_frozen_field_20260918')/scene;r=root/'reconstruction/field';out=OUT/scene;out.mkdir();info=json.loads((r/'teachers.json').read_text());auth=json.loads((r/'responsibility.json').read_text());assert sha256_file(r/'responsibility.json')==info['authority_sha256'];rec=info['teacher_files']['siglip'];desc=F.normalize(load(r/rec['path'],rec['sha256'])['descriptors'].float(),dim=1);ep=load(r/'episodes.pt',json.loads((r/'episodes.json').read_text())['sha256']);br=json.loads((r/'pixel_evidence96/report.json').read_text());bp=load(r/'pixel_evidence96/field.pt',br['field_sha256']);state=bp['state_dict'];basis=state['native_decoder.weight'];assert basis.shape[1]==64;observed=state['observed'];n=len(observed)
 train=torch.tensor([info['frames'][int(v)] in info['train_frames'] for v in ep['proposal_views']]);source=desc[train].double();_,vectors=torch.linalg.eigh(source.T@source);basis192=vectors[:,-192:].float();del source,vectors
 compact=desc@basis;wide=desc@basis192;areas=ep['masks'].float().flatten(1).mean(1);stratum=(areas>.01).long()+(areas>.1).long();num=torch.zeros(3,n,64);mass=torch.zeros(3,n);wide_num=torch.zeros(n,192)
 for frame in info['train_frames']:
  vi=info['frames'].index(frame);ids=torch.where(ep['proposal_views']==vi)[0];rec=auth['views'][vi];hit=load(r/rec['relative_path'],rec['sha256']);weights,_=source_pixel_evidence_weights(ep['masks'][ids].flatten(1),hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float(),n,partition_overlap=True)
  wide_num+=weights.T@wide[ids]
  for b in range(3):
   use=stratum[ids]==b
   if use.any():num[b]+=weights[use].T@compact[ids[use]];mass[b]+=weights[use].sum(0)
  del weights,hit
 total=mass.sum(0);pooled=num.sum(0)/total[:,None].clamp_min(1e-8);error=float((pooled[observed]-state['latent'][observed,:64]).abs().max());assert error<2e-4,error;slots=num/mass[...,None].clamp_min(1e-8);single192=wide_num/total[:,None].clamp_min(1e-8);bank=load(root/'official_protocol_evaluation/queries.pt');classes=OPENGAUSSIAN_NYU40_CLASS_SPLITS['19'];q=F.normalize(bank['encoding'][[bank['texts'].index(NYU40_ID_TO_NAME[c]) for c in classes]].float(),dim=1)
 raw=F.normalize(pooled@basis.T,dim=1)@q.T;slot_scores=F.normalize(slots,dim=2)@(basis.T@q.T);slot_scores=slot_scores.masked_fill((mass<=0)[...,None],-torch.inf);stratified=slot_scores.amax(0);broad=F.normalize(single192,dim=1)@(basis192.T@q.T);prediction={k:torch.where(observed,torch.tensor(classes)[v.argmax(1)],0).numpy() for k,v in [('single64',raw),('area_strata_3x64',stratified),('single192',broad)]};original=np.load(root/'official_protocol_evaluation/predictions.npz')['19'];assert np.array_equal(prediction['single64'],original);np.savez_compressed(out/'predictions.npz',**prediction)
 torch.save(dict(geometry_binding=bp['geometry_binding'],native_basis64=basis,native_area_strata=slots,support_mass=mass,observed=observed,native_basis192=basis192,native_single192=single192),out/'native_field_control.pt')
 gr=json.loads((root/'geometry_report.json').read_text())['results']['independent_scale'];geo=FrozenGaussianGeometry.load(gr['geometry'],expected_sha256=gr['geometry_sha256'],device='cpu');geo.assert_binding(bp['geometry_binding']);xyz,labels=_read_label_ply(Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/(scene+'_vh_clean_2.labels.ply'));gt,_=assign_vala_pseudo_labels(geo.xyz.numpy(),geo.scales.numpy(),geo.quaternions.numpy(),xyz,labels,radius_factor=5.,candidate_k=1000,fallback_k=1,class_balance=True,chunk_size=512);weight=(geo.scales.prod(1)*geo.opacities.flatten()).numpy();metrics={k:volume_weighted_split_metrics(gt,v,weight,classes) for k,v in prediction.items()}
 result=dict(scene=scene,diagnostic_only=True,source_only_construction=True,selection_before_gt=True,fusion_replay_max_error=error,native_prediction_replay_exact=True,three_strata_are_not_physical_parts=True,region_counts_by_stratum=[int(((stratum==i)&train).sum()) for i in range(3)],metrics=metrics,registration_sha256=sha256_file(OUT/'registration.json'),field_sha256=sha256_file(out/'native_field_control.pt'),prediction_sha256=sha256_file(out/'predictions.npz'),teacher_sha256=info['teacher_files']['siglip']['sha256']);(out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(scene,{k:v['miou'] for k,v in metrics.items()},flush=True);return dict(report_sha256=sha256_file(out/'report.json'),miou={k:v['miou'] for k,v in metrics.items()})
save()
with torch.no_grad():
 while len(record['scenes'])+len(record['failures'])<len(SCENES):
  ready=False
  for scene in SCENES:
   if scene in record['scenes'] or any(r['scene']==scene for r in record['failures']):continue
   if not (Path('output/scannet_frozen_field_20260918')/scene/'official_protocol_evaluation/report.json').exists():continue
   ready=True
   try:record['scenes'][scene]=run(scene)
   except Exception as err:record['failures'].append(dict(scene=scene,error=repr(err)))
   save()
  if not ready:
   if not any(x['alive'] for x in json.loads(Path('output/benchmark_closure_20260919/liveness.json').read_text())['workers']):break
   time.sleep(30)
record['complete']=len(record['scenes'])==len(SCENES) and not record['failures'];save()
