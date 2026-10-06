"""Actual-field opt-in module replay; no GT and no new fitting."""
import json
from pathlib import Path
import numpy as np
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS,NYU40_ID_TO_NAME
from radio_gs.querying.bounded_native_scope import refine_native_scope_logits
from radio_gs.utils.immutable_artifacts import sha256_file
torch.set_num_threads(4);out=Path('output/scannet_system_refinement_20260919_v2/module_replay.json');assert not out.exists();scene='scene0000_00';root=Path('output/scannet_frozen_field_20260918')/scene;fp=root/'reconstruction/field/bounded_fallback_admission/field.pt';report=json.loads(fp.with_name('report.json').read_text());assert sha256_file(fp)==report['field_sha256'];p=torch.load(fp,map_location='cpu',weights_only=False);model=ElementRetrievalField(**p['architecture']).cuda();model.load_state_dict(p['state_dict']);bank=torch.load(root/'official_protocol_evaluation/queries.pt',map_location='cpu',weights_only=False);classes=sorted(set(sum([SPLITS[s] for s in ['19','15','10']],[])));q=bank['encoding'][[bank['texts'].index(NYU40_ID_TO_NAME[c]) for c in classes]].float().cuda()
with torch.no_grad():
 n=model.readout_log_scale['text'].clamp(-4,6).exp()*(model.native_cosine(q)-model.native_query_thresholds['text'](q));s=model.logits(q,'text');candidate=refine_native_scope_logits(n,s,max_correction=1.);reference=n+(s-n).clamp(-1,1);assert torch.equal(candidate,reference);assert refine_native_scope_logits(n,s) is n;predictions={};differences={}
 for split in ['19','15','10']:
  ids=SPLITS[split];ix=[classes.index(c) for c in ids];pred=torch.tensor(ids,device='cuda:0')[candidate[:,ix].argmax(1)];pred=torch.where(model.observed,pred,0).cpu().numpy();old=np.load(out.parent/scene/'bounded_logit.npz')[split];differences[split]=int((pred!=old).sum());assert differences[split]==0
result=dict(scene=scene,actual_field_recomputed=True,logit_max_difference=float((candidate-reference).abs().max()),per_split_prediction_mismatch=differences,default_identity=True,benchmark_gt_opened=False,field_sha256=report['field_sha256'],module_sha256=sha256_file('radio_gs/querying/bounded_native_scope.py'),implementation_sha256=sha256_file(__file__),coverage='Actual0000 field, every element and all3 class splits; full8 high-margin audit is separate.')
out.write_text(json.dumps(result,indent=2)+'\n');print(result)
