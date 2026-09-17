"""Measure view grouping and recency in the actual training episode order."""
import argparse,json,hashlib
from itertools import groupby
from collections import Counter
from pathlib import Path
import torch
from radio_gs.training.retrieval_supervision import episode_order
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--seed',type=int,default=1000);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 receipt=json.loads((a.root/'episodes.json').read_text());assert sha256_file(a.root/'episodes.pt')==receipt['sha256']
 d=torch.load(a.root/'episodes.pt',weights_only=False);results={}
 for shuffle in [False,True]:
  results[str(shuffle)]={}
  for index,modality in enumerate(['text','image','prompt']):
   items=[e for e in d['episodes'] if e['split']=='train' and e['modality']==modality]
   count=max(0,(a.steps+2-index)//3);ids=[episode_order(len(items),i//len(items),a.seed+index,shuffle)[i%len(items)] for i in range(count)]
   views=[items[i]['target_view'] for i in ids];runs=[len(list(group)) for _,group in groupby(views)];counts=Counter(views)
   results[str(shuffle)][modality]={'updates':count,'max_consecutive_same_target_view':max(runs),'mean_consecutive_same_target_view':sum(runs)/len(runs),'target_view_updates':dict(counts),'steps_since_last_view_update':{v:len(views)-1-max(i for i,x in enumerate(views) if x==v) for v in counts},'schedule_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
 a.output.write_text(json.dumps({'episode_sha256':receipt['sha256'],'steps':a.steps,'seed':a.seed,'schedules':results},indent=2)+'\n')
 print({k:{m:v['max_consecutive_same_target_view'] for m,v in x.items()} for k,x in results.items()})
if __name__=='__main__':main()
