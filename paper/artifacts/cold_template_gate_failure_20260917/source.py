import json,torch
from pathlib import Path
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
base=Path('output/lerf_view_balanced_rebuild_20260917/figurines/field')
f=torch.load(base/'cold_query_fixture.pt',weights_only=False)
p=torch.load(base/'exact_render_query_threshold96/field.pt',weights_only=False)
m=ElementRetrievalField(**p['architecture']).cuda(0);m.load_state_dict(p['state_dict']);m.requires_grad_(False)
e=NativeThreeQueryEncoder(f['encoder_identity'],'cuda:0');q=e.encode_text([f['text']]).half().float();r=f['expected_encodings']['text'].float().cuda(0)
print('query',f['text'],flush=True)
with torch.no_grad():
 rows=[]
 for name,x in [('online',q),('cached',r)]:
  h=m.grounded_transfer;index,accepted,score=h.route(x);threshold=m.native_query_thresholds['text'](x);cos=m.native_cosine(x);supp=cos>threshold;overlap=h.support_overlap(m.latent,supp,m.observed);v,i=overlap.topk(3,dim=1)
  tq=F.normalize(x,dim=-1);ev=torch.stack([(tq*F.normalize(h.prototypes[index],dim=-1)).sum(-1),(tq*h.routing_scope_images[index]).sum(-1)],-1)@h.routing_scope_weights+h.routing_scope_bias
  s=m.retrieve(x,'text')['scores'];rows.append(s)
  print(name,dict(route=index.tolist(),accepted=accepted.tolist(),score=score.tolist(),acceptance=h.acceptance[index].tolist(),scope_evidence=ev.tolist(),threshold=threshold.tolist(),support=int(supp.sum()),overlap_scores=v.tolist(),overlap_ids=i.tolist()),flush=True)
 print('maxdelta',float((rows[0]-rows[1]).abs().max()),'changed_memberships',int(((rows[0]>.5)!=(rows[1]>.5)).sum()),flush=True)
