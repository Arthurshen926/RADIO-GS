"""Audit immutable source region pairs and export text-review evidence."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.training.correspondence_audit import audit_correspondence_graph
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 receipt=json.loads((a.root/'episodes.json').read_text())
 if sha256_file(a.root/'episodes.pt')!=receipt['sha256']:raise ValueError('episode identity')
 data=torch.load(a.root/'episodes.pt',weights_only=False)
 image=F.normalize(data['queries']['image'].float(),dim=-1);text=F.normalize(data['queries']['text'].float(),dim=-1)
 scores=image@text.T;top=scores.topk(2,dim=-1);words=data['queries']['queries']
 torch.set_num_threads(4)
 graph=audit_correspondence_graph(data['episodes'],data['proposal_views'])
 pairs=sorted({(e['split'],min(e['query_region'],e['target_region']),max(e['query_region'],e['target_region'])) for e in data['episodes'] if e['modality']=='image'})
 evidence=[]
 for split,left,right in pairs:
  labels=top.indices[[left,right],0].tolist();margins=(top.values[[left,right],0]-top.values[[left,right],1]).tolist()
  evidence.append({'split':split,'regions':[left,right],'views':data['proposal_views'][[left,right]].tolist(),
                   'top_text':[words[x] for x in labels],'top_margin':margins,'text_agrees':labels[0]==labels[1],
                   'native_image_cosine':float(image[left]@image[right]),'physical_identity_ground_truth':None})
 grouped=defaultdict(list)
 for episode in data['episodes']:
  if episode['split']=='train':grouped[(episode['modality'],episode.get('text_index',episode['query_region']))].append(episode)
 probability=data['region_probability'];conflicts=[]
 for (modality,query_id),episodes in grouped.items():
  positive=torch.zeros(probability.shape[1],dtype=torch.bool);negative=positive.clone()
  for episode in episodes:
   pos=probability[episode.get('positive_regions',[episode['target_region']])].amax(0)>.7
   neg=probability[episode['negative_regions']].amax(0)>.7 if episode['negative_regions'] else torch.zeros_like(pos)
   known=pos^neg;positive|=pos&known;negative|=neg&known
  overlap=positive&negative
  if overlap.any():conflicts.append({'modality':modality,'query_id':query_id,'episodes':len(episodes),'conflicting_rows':int(overlap.sum()),'labelled_rows':int((positive|negative).sum())})
 result={'schema':'source_region_identity_audit.v1','episode_sha256':receipt['sha256'],'graph':graph,'pairs':evidence,
         'cross_view_training_label_conflicts':conflicts,'text_disagreement_pairs':sum(not x['text_agrees'] for x in evidence),'total_pairs':len(evidence),
         'semantic_ground_truth':False,'interpretation':'track consistency and teacher agreement are diagnostics, not proof of identity'}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['graph','pairs']},indent=2))
if __name__=='__main__':main()
