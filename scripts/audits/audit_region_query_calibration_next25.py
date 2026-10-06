"""Source-only audit of native region-query calibration and phrase separation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
 root=Path('output/full_field_rebuild_next19')/a.scene/'reconstruction/field';capfile=root/'caption_region_evidence.pt';caprec=json.loads(capfile.with_suffix('.json').read_text());assert sha256_file(capfile)==caprec['sha256']
 cap=torch.load(capfile,map_location='cpu',weights_only=False);fieldpath=root/'bounded_fallback_admission/field.pt';fr=json.loads((fieldpath.parent/'report.json').read_text());assert sha256_file(fieldpath)==fr['field_sha256'];payload=torch.load(fieldpath,map_location='cpu',weights_only=False)
 model=ElementRetrievalField(**payload['architecture']);model.load_state_dict(payload['state_dict']);model.requires_grad_(False)
 q=F.normalize(cap['text_embeddings'].float(),dim=-1);d=F.normalize(cap['native_descriptors'].float(),dim=-1);sim=d@q.T;th=model.native_query_thresholds['text'](q);positive=[];ranks=[];best_competing=[]
 for i,rec in enumerate(cap['records']):
  ids=rec['phrase_indices'];assert ids
  own=sim[i,ids];j=ids[int(own.argmax())];positive.append((float(own.max()),float(th[j])))
  tmp=sim[i].clone();tmp[ids]=-2
  ranks.append(int((tmp>own.max()).sum()))
  best_competing.append(float(tmp.max()))
 pscore=np.array(positive);other=np.array(best_competing)
 result=dict(scene=a.scene,complete=True,source_only=True,benchmark_labels_opened=False,regions=len(sim),phrases=len(q),positive_above_source_element_threshold=float(np.mean(pscore[:,0]>pscore[:,1])),positive_top1_among_all_source_phrases=float(np.mean(np.array(ranks)==0)),positive_cosine_percentiles=np.percentile(pscore[:,0],[5,25,50,75,95]).tolist(),source_threshold_percentiles=np.percentile(pscore[:,1],[5,25,50,75,95]).tolist(),best_nonprompted_cosine_percentiles=np.percentile(other,[5,25,50,75,95]).tolist(),field_sha256=fr['field_sha256'],caption_sha256=caprec['sha256'],implementation_sha256=sha256_file(__file__))
 out=Path('output/query_region_readout_next25')/a.scene/'source_calibration.json';out.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if 'percentiles' in k or k.startswith('positive_')},flush=True)
if __name__=='__main__':main()
