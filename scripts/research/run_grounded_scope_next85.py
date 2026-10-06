"""Fixed-source instruction verifier followed by matched scope reconstruction."""
import argparse,json,os,subprocess
from pathlib import Path
import torch

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);p.add_argument('--seed-base',default='output/grounded_seeds_next85');p.add_argument('--track-base',default='output/grounded_tracks_next85');p.add_argument('--output-base',default='output/grounded_scope_next85');p.add_argument('--initializations',nargs='+',default=['semantic','source']);a=p.parse_args();key='lerf_'+a.scene;seedbase=Path(a.seed_base);trackbase=Path(a.track_base);root=Path(a.output_base);root.mkdir(exist_ok=True);(trackbase/key).mkdir(parents=True,exist_ok=True);rows=json.loads((seedbase/key/'proposals.json').read_text());ids=[str(r['query_index']) for r in rows];env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);py='/root/miniconda3/envs/cybersim_agent/bin/python';reuse=[]
for qi in ids:
 seed=load(seedbase/key/f'{qi}.pt');destination=trackbase/key/f'{qi}.pt'
 if destination.exists():continue
 for sb,tb in [('output/region_seeds_next80','output/region_seed_tracks_next80'),('output/content_seeds_next83','output/content_tracks_next83'),('output/counterfactual_seeds_next84/raw','output/counterfactual_tracks_next84/raw'),('output/counterfactual_seeds_next84/contrast','output/counterfactual_tracks_next84/contrast')]:
  previous=load(Path(sb)/key/f'{qi}.pt')
  if previous['seed_frame']==seed['seed_frame'] and torch.equal(previous['masks'],seed['masks']):
   tp=Path(tb)/key/f'{qi}.pt';track=load(tp);track['seed_evidence']=seed;track['tracking_reused_from']=str(tp);torch.save(track,destination);reuse.append(dict(query_index=int(qi),source=str(tp)));break
(root/(key+'_reuse.json')).write_text(json.dumps(reuse,indent=2)+'\n')
with (root/(key+'_tracking.log')).open('w') as log:
 subprocess.run(['/root/miniconda3/envs/sam3/bin/python','scripts/research/track_full_text_sources_next79.py','--key',key,'--query-indices',*ids,'--seed-base',str(seedbase),'--output-base',str(trackbase)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
for initialization in a.initializations:
 out=root/initialization/key;out.mkdir(parents=True,exist_ok=True);cmd=[py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base','output/native_write_next75/all/calibrated','--cohort','all','--evidence-base','output/typed_evidence_next44','--output-base',str(out.parent),'--mode','seeded','--tracks-base',str(trackbase),'--retained-selection',str(Path('output/region_seed_positive_next80')/key/'typed_readout.pt'),'--extra-track-frames','--regional-positive-evidence','--scope-initialization',initialization,'--query-indices',*ids]
 with (out/'execution.log').open('w') as log:
  subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run([py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(out.parent),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
