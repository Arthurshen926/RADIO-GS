"""Describe actual full-source tracking coverage and unchanged seed selection."""
import json
from pathlib import Path
import torch
result={}
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 root=Path('output/full_text_tracks_next79')/('lerf_'+scene);contract=json.loads((root/'contract_all_0_of_2.json').read_text());rows=[]
 for qi,q in enumerate(contract['texts']):
  new=torch.load(root/f'{qi}.pt',map_location='cpu',weights_only=False);old=torch.load(Path('output/text_video_next63/tracks')/('lerf_'+scene)/f'{qi}.pt',map_location='cpu',weights_only=False)
  rows.append(dict(query=q,old_frames=len(old['frames']),new_frames=len(new['frames']),same_seed=old['seed_frame']==new['seed_frame'],seed_frame=new['seed_frame'],present_frames=sum(bool(v['present']) for v in new['frames'].values())))
 result[scene]=rows
 print(scene,len(rows),'unchanged_seeds',all(r['same_seed'] for r in rows),'unseeded',[r['query'] for r in rows if r['seed_frame'] is None])
Path('output/next79/tracking_coverage.json').write_text(json.dumps(result,indent=2)+'\n')
