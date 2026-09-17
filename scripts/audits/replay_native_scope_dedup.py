"""Bounded source-only replay of coarse/native mask identity collisions."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from radio_gs.utils.native_scope_identity import native_scope_identity
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model,set_requested_cuda_device,sam3_autocast_context
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';out=root.parent/'native_scope_dedup_replay_20260918';out.mkdir(exist_ok=False);snap=out/'sources';snap.mkdir();impl={}
for f in [Path(__file__),Path('radio_gs/utils/native_scope_identity.py'),Path('scripts/audits/ground_source_caption_phrases.py')]:shutil.copy2(f,snap/f.name);impl[str(f)]=sha256_file(f)
info=json.loads((root/'teachers.json').read_text());source=json.loads((root/'source_frames.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];gp=root/'caption_grounded_regions.pt';assert sha256_file(gp)==json.loads(gp.with_suffix('.json').read_text())['sha256'];g=torch.load(gp,weights_only=False);assert g['teacher_manifest_sha256']==sha256_file(root/'teachers.json');frame=info['train_frames'][0];vi=info['frames'].index(frame);records=sorted([r for r in g['records'] if r['frame']==frame],key=lambda r:-len(set(r['phrase_indices'])));qids=[]
for r in records:
 for qi in sorted(set(r['phrase_indices'])):
  if qi not in qids:qids.append(qi)
  if len(qids)==12:break
 if len(qids)==12:break
src=next(s for s in source['frames'] if s['frame_idx']==frame);imagepath=Path(source['image_dir'])/src['source_file'];assert sha256_file(imagepath)==src['source_sha256'];image=Image.open(imagepath).convert('RGB');h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width'];checkpoint=Path('checkpoints/sam3_modelscope/sam3.pt');assert sha256_file(checkpoint)==g['checkpoint_sha256'];set_requested_cuda_device(a.device);model=_load_sam3_model(checkpoint_path=str(checkpoint),device=a.device,confidence_threshold=.7,dtype='float32',resolution=1008,build_on_cpu=True);observations=[];old={};new={};coarse_variants={}
with torch.inference_mode():
 with sam3_autocast_context(a.device,torch.bfloat16):state=model.set_image(image)
 for qi in qids:
  model.reset_all_prompts(state)
  with sam3_autocast_context(a.device,torch.bfloat16):prediction=model.set_text_prompt(g['phrases'][qi],state)
  for prob,confidence in zip(prediction['masks_logits'],prediction['scores']):
   prob=prob.float().squeeze();assert prob.shape==(image.height,image.width);native=prob>.5;small=F.interpolate(prob[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0]>.5
   if int(small.sum())<2 or float(small.float().mean())>.9:continue
   packed=np.packbits(native.cpu().numpy(),axis=-1);coarse=small.cpu().numpy();legacy=coarse.tobytes();identity=native_scope_identity(coarse,packed,[image.height,image.width]);record=dict(phrase_index=qi,phrase=g['phrases'][qi],confidence=float(confidence),native_identity=identity[-1],packed_native=torch.from_numpy(packed),coarse_mask=torch.from_numpy(coarse));idx=len(observations);observations.append(record);coarse_variants.setdefault(legacy,set()).add(identity)
   for key,inventory in [(legacy,old),(identity,new)]:
    if key not in inventory:inventory[key]=dict(indices=[],selected=idx)
    inventory[key]['indices'].append(idx)
    if record['confidence']>observations[inventory[key]['selected']]['confidence']:inventory[key]['selected']=idx
  print(a.scene,frame,g['phrases'][qi],len(observations),flush=True)
errors={};examples=[]
for name,inventory in [('coarse_only',old),('native_and_coarse',new)]:
 wrong=0
 for entry in inventory.values():
  selected=observations[entry['selected']]
  for i in entry['indices']:
   original=observations[i]
   if original['native_identity']!=selected['native_identity']:
    wrong+=1
    if name=='coarse_only':
     left=np.unpackbits(original['packed_native'].numpy(),axis=-1)[:,:image.width].astype(bool);right=np.unpackbits(selected['packed_native'].numpy(),axis=-1)[:,:image.width].astype(bool);examples.append(dict(original_phrase=original['phrase'],retained_phrase=selected['phrase'],native_iou=float((left&right).sum()/max((left|right).sum(),1)),original_identity=original['native_identity'],retained_identity=selected['native_identity']))
 errors[name]=wrong
assert errors['native_and_coarse']==0
artifact=out/'replay.pt';torch.save(dict(observations=observations,native_shape=[image.height,image.width],source_only=True,benchmark_data_used=False),artifact)
report=dict(schema='native_scope_dedup_replay.v1',scene=a.scene,frame=frame,source_only=True,benchmark_data_used=False,selection='first training frame, first 12 distinct phrases in descending existing coarse-mask alias count; no real query or label selection',query_indices=qids,queries=[g['phrases'][i] for i in qids],accepted_detections=len(observations),legacy_regions=len(old),corrected_regions=len(new),coarse_keys_with_multiple_native_scopes=sum(len(v)>1 for v in coarse_variants.values()),native_scope_assignment_errors=errors,examples=examples,inputs={str(gp):sha256_file(gp),str(imagepath):sha256_file(imagepath),str(checkpoint):sha256_file(checkpoint)},replay_sha256=sha256_file(artifact),implementations=impl,limitation='One deliberately alias-rich source frame per scene and 12 phrases; not prevalence or final retrieval improvement. Existing full caches/checkpoints remain unchanged; they require regeneration to consume v2.')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,{k:report[k] for k in ['accepted_detections','legacy_regions','corrected_regions','coarse_keys_with_multiple_native_scopes','native_scope_assignment_errors']},flush=True)
