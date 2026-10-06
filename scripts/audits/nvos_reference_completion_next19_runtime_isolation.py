"""Real same-map horns A/B/A runtime isolation; no target GT/RGB access."""
import json,os,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.prompt_transfer_20260919_evaluate import refine as prepare_base
from scripts.audits.nvos_reference_completion_next19_runtime import complete_reference_query,load_bound_query_state

torch.set_num_threads(4);device='cuda:0';out=Path('output/nvos_reference_completion_next19_runtime_isolation');out.mkdir(exist_ok=False)
manifest=json.loads(Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json').read_text());tasks=['horns_center','horns_left'];scenes={t:next(s for s in manifest['scenes'] if s['scene_id']==t) for t in tasks}
forbidden=set()
for s in scenes.values():
 forbidden.add(str(Path(s['prompt']['negative_path']).resolve()))
 for f in s['frames']:
  if f['frame_id'] in s['evaluation_frame_ids']:
   for k in ['rgb_path','annotation_rgb_path','gt_mask_path']:forbidden.add(str(Path(f[k]).resolve()))
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)) and str(Path(os.fsdecode(args[0])).resolve()) in forbidden:raise RuntimeError('No target or negative assets permitted in isolation check')
sys.addaudithook(guard)
root=Path('output/nvos_ropefix_pipeline_20260917/horns_center');gr=json.loads((root/'frozen_geometry/report.json').read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=device);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
identity=json.loads((root/'field/native_query_encoder.json').read_text());encoder=NativeThreeQueryEncoder(identity,device);queries={};sources={}
for t in tasks:
 qp=Path('output/nvos_positive_queries_20260918')/t/'receipt.json';qr=json.loads(qp.read_text());assert not qr['negative_scribble_opened'];assert sha256_file(qr['image_query_crop']['path'])==qr['image_query_crop']['sha256'];assert sha256_file(qr['mask'])==qr['mask_sha256']
 queries[t]=encoder.encode_image(Image.open(qr['image_query_crop']['path']).convert('RGB'));sources[t]=(qr,qp)
encoder.vision=None
import gc
gc.collect();torch.cuda.empty_cache()
run='image_supervised_native_residual_measured_support512';fr=json.loads((root/'field'/run/'report.json').read_text());field,payload=load_element_retrieval_field(root/'field'/run/'field.pt',expected_sha256=fr['field_sha256'],geometry=geo,expected_query_encoder=identity,device=device)
original_latent=field.latent.detach().clone();assets={};bindingkeys={'field_sha256','geometry_sha256','positive_sha256','sam_mask_sha256','query_receipt_sha256','parameters_sha256','stage'}
with torch.no_grad():
 for t in tasks:
  qr,qp=sources[t];width,height=qr['prompt_raster_wh'];ref=next(f for f in scenes[t]['frames'] if f['frame_id']==scenes[t]['prompt']['frame_id']);camera=prior.cameras[int(ref['canonical_index'])]
  def mask(path):
   x=np.asarray(Image.open(path));return torch.from_numpy(np.any(x>127,axis=2) if x.ndim==3 else x>127)
  pos=mask(qr['official_scribbles']['positive']['path']);sam=mask(qr['mask']);assert pos.shape==sam.shape==(height,width)
  base,record=prepare_base(field.retrieve(queries[t],'image')['scores'],field.observed_for('image'),geo,camera,prior,width,height,pos,sam,'prompt_sam')
  p=out/(t+'_prompt_sam_elements.pt');torch.save(dict(scores=base.cpu()),p)
  binding=dict(field_sha256=fr['field_sha256'],geometry_sha256=g['geometry_sha256'],positive_sha256=qr['official_scribbles']['positive']['sha256'],sam_mask_sha256=qr['mask_sha256'],query_receipt_sha256=sha256_file(qp),parameters_sha256=sha256_file('output/nvos_reference_completion_next19/registration.json'),stage='prompt_sam',state_sha256=sha256_file(p))
  Path(str(p)+'.binding.json').write_text(json.dumps(binding,indent=2)+'\n');loaded,_=load_bound_query_state(p,{k:binding[k] for k in bindingkeys},device=device)
  assets[t]=dict(scores=loaded,observed=field.observed_for('image'),geo=geo,camera=camera,prior=prior,width=width,height=height,positive=pos,sam=sam,mode='reference_core_component',field=field,base_stage='prompt_sam')
 a_before=assets[tasks[0]]['scores'].clone();outputs=[];records=[]
 for t in [tasks[0],tasks[1],tasks[0]]:
  value,record=complete_reference_query(**assets[t]);outputs.append(value.cpu());records.append(record)
 assert torch.equal(outputs[0],outputs[2]),'A/B/A output changed'
 assert torch.equal(a_before,assets[tasks[0]]['scores']),'Input state mutated'
 assert torch.equal(original_latent,field.latent),'Persistent field changed'
 comparisons={}
 for i,t in enumerate(tasks):
  old=torch.load(Path('output/nvos_reference_completion_next19')/t/'reference_core_component_elements.pt',map_location='cpu',weights_only=False)['scores'];err=float((outputs[i]-old).abs().max());assert err<=1e-6
  comparisons[t]=dict(max_abs_element_score_error=err,selected_element_mismatches=int(((outputs[i]>=.5)!=(old>=.5)).sum()))
 result=dict(schema='next19_real_query_batch_isolation.v1',sequence=[tasks[0],tasks[1],tasks[0]],a_b_a_scores_bitwise_equal=True,input_base_scores_unchanged=True,persistent_latent_unchanged=True,target_assets_read=False,comparisons=comparisons,runtime_sha256=sha256_file('scripts/audits/nvos_reference_completion_next19_runtime.py'),input_bindings={str(p):sha256_file(p) for p in out.glob('*.binding.json')})
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
