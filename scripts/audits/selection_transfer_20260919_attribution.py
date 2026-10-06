"""Risk transfer confusion and same-phrase cross-view consistency; no tuning."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/selection_transfer_20260919');out=root/'attribution.json';assert not out.exists();report={}
def confusion(truth,pred):
 tp=int((truth&pred).sum());fp=int((~truth&pred).sum());fn=int((truth&~pred).sum());tn=int((~truth&~pred).sum())
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,accuracy=(tp+tn)/len(truth),always_safe_accuracy=float((~truth).float().mean()))
for scene in ['figurines','ramen']:
 fit=json.load(open(Path('output/source_selected_contract_20260919')/scene/'report.json'));hold=json.load(open(root/f'source_holdout_{scene}'/'report.json'));e=torch.load(Path('output/depth_opacity_pipeline_20260918')/scene/'reconstruction/field/caption_region_evidence.pt',map_location='cpu',weights_only=False);aa=defaultdict(list);bb=defaultdict(list)
 for r in fit['observations']:aa[r['query_index']].append(r)
 for r in hold['observations']:bb[r['query_index']].append(r)
 def delta(group):return [float(np.mean([r['metrics']['final'][m]-r['metrics']['native'][m] for r in group])) for m in ['probability_iou','selected_iou']]
 ids=sorted(aa);ds=torch.tensor([delta(aa[i]) for i in ids]);vec=F.normalize(e['text_embeddings'].float(),dim=1);s=vec[ids]@vec[ids].T;s.fill_diagonal_(-2);nn=s.argmax(1);arms={}
 for arm in ['joint','selected']:
  harm=(ds<0).all(1) if arm=='joint' else ds[:,1]<0;hids=sorted(bb);hd=torch.tensor([delta(bb[i]) for i in hids]);truth=(hd<0).all(1) if arm=='joint' else hd[:,1]<0;prediction=harm[(vec[hids]@vec[ids].T).argmax(1)]
  arms[arm]=dict(leave_phrase_out=confusion(harm,harm[nn]),view_disjoint_source_phrase=confusion(truth,prediction))
 shared=[dict(phrase=e['phrases'][i],fit_delta=delta(aa[i]),holdout_delta=delta(bb[i]),selected_sign_flip=(delta(aa[i])[1]<0)!=(delta(bb[i])[1]<0)) for i in sorted(set(aa)&set(bb))]
 report[scene]=dict(confusions=arms,shared_phrases=len(shared),shared_selected_sign_flips=sum(r['selected_sign_flip'] for r in shared),shared_rows=shared)
result=dict(results=report,implementation_sha256=sha256_file(__file__),summary_sha256=sha256_file(root/'summary.json'),interpretation='Gate views are disjoint; parent extent saw both. Majority-safe accuracy is not evidence of harmful-range detection. Pseudo scopes do not define hidden physical identity.')
out.write_text(json.dumps(result,indent=2)+'\n')
for scene,r in report.items():print(scene,r['confusions'])
