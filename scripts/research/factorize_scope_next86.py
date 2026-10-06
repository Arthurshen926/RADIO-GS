"""Same admitted hypotheses, independent memberships then text-set union."""
import argparse,json,os,subprocess,shutil
from pathlib import Path
import torch

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene;rows=json.loads((Path('output/multi_seeds_next86')/key/'proposals.json').read_text());ids=[str(r['query_index']) for r in rows if r['secondary_rank'] is not None];root=Path('output/multi_scope_next86');out=root/'factorized'/key;out.mkdir(parents=True,exist_ok=True);py='/root/miniconda3/envs/cybersim_agent/bin/python';env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);baseline=Path('output/grounded_scope_next85/source')/key/'typed_readout.pt'
for name in ['field.pt','calibration.pt']:shutil.copyfile(baseline.parent/name,out/name)
with (out/'execution.log').open('w') as log:
 if ids:
  subprocess.run([py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base','output/native_write_next75/all/calibrated','--cohort','all','--evidence-base','output/typed_evidence_next44','--output-base',str(root/'secondary'),'--mode','seeded','--tracks-base','output/secondary_tracks_next86','--retained-selection',str(baseline),'--extra-track-frames','--regional-positive-evidence','--scope-initialization','source','--query-indices',*ids],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  result=load(baseline);secondary=load(root/'secondary'/key/'typed_readout.pt');indices=[int(i) for i in ids];result['selection_scores'][:,indices]=torch.maximum(result['selection_scores'][:,indices],secondary['selection_scores'][:,indices]);result['text_union_factorization']=dict(primary=str(baseline),secondary=str(root/'secondary'/key/'typed_readout.pt'),query_indices=indices,rule='pointwise max of independently decoded memberships; hard selection equals set union',extra_decoder_work=True);torch.save(result,out/'typed_readout.pt')
 else:
  result=load(baseline);torch.save(result,out/'typed_readout.pt')
 subprocess.run([py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(out.parent),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print(key,'factorized complete',flush=True)
