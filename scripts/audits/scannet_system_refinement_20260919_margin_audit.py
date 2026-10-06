"""Verify bounded structural correction never overrides native margins above 2."""
import json
from pathlib import Path
import numpy as np
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS,NYU40_ID_TO_NAME
from radio_gs.utils.immutable_artifacts import sha256_file
torch.set_num_threads(4);root=Path('output/scannet_system_refinement_20260919_v2');out=root/'margin_audit.json';assert not out.exists();rows={}
with torch.no_grad():
 for scene in ['scene0000_00','scene0062_00','scene0070_00','scene0097_00','scene0140_00','scene0347_00','scene0400_00','scene0590_00']:
  if not (root/scene/'report.json').exists():raise RuntimeError(scene+' evaluation incomplete')
  r=Path('output/scannet_frozen_field_20260918')/scene;fp=r/'reconstruction/field/bounded_fallback_admission/field.pt';p=torch.load(fp,map_location='cpu',weights_only=False);m=ElementRetrievalField(**p['architecture']);m.load_state_dict(p['state_dict']);b=torch.load(r/'official_protocol_evaluation/queries.pt',map_location='cpu',weights_only=False);rows[scene]={};cal=np.load(root/scene/'calibrated_native.npz');bound=np.load(root/scene/'bounded_logit.npz')
  for split in ['19','15','10']:
   ids=SPLITS[split];q=b['encoding'][[b['texts'].index(NYU40_ID_TO_NAME[c]) for c in ids]].float();nl=m.readout_log_scale['text'].clamp(-4,6).exp()*(m.native_cosine(q)-m.native_query_thresholds['text'](q));values=nl.topk(2,dim=1).values;margin=values[:,0]-values[:,1];protected=(m.observed&(margin>2+1e-5)).numpy();changed=cal[split]!=bound[split];violations=int((changed&protected).sum());assert violations==0
   rows[scene][split]=dict(observed=int(m.observed.sum()),changed=int(changed.sum()),protected_high_margin=int(protected.sum()),protected_violations=violations)
result=dict(relative_to='Existing calibrated native logits; not raw cosine ordering',bound=1.,guarantee='Each class changes by at most1; original margin>2 is preserved. CPU check uses2+1e-5 tolerance.',rows=rows,implementation_sha256=sha256_file(__file__));out.write_text(json.dumps(result,indent=2)+'\n');print('margin audit complete')
