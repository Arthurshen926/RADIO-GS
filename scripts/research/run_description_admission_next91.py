"""Source-only secondary identity admission, without re-certifying the parent.
The initial all-four-view-A control abstains everywhere; record both rules.
"""
import argparse,json,os,shutil,subprocess
from pathlib import Path
import torch
from scripts.research.multi_scope_next86 import merge
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene;rows=json.loads((Path('output/multi_seeds_next89')/key/'proposals.json').read_text());decisions=Path('output/next91')/key/'joint_identity.json';judgments=json.loads(decisions.read_text())['results'] if decisions.exists() else [];admitted={r['query_index'] for r in judgments if all(j['answer']=='A' for j in r['comparison'] if j['role']=='secondary')};ids=[]
for row in rows:
 if row['query_index'] not in admitted:row['secondary_rank']=None
 elif row['secondary_rank'] is not None:ids.append(str(row['query_index']))
tracks='output/description_tracks_next91';merge(key,rows,'output/grounded_tracks_next88','output/secondary_tracks_next89',tracks);out=Path('output/description_scope_next91')/key;out.mkdir(parents=True,exist_ok=True);baseline=Path('output/grounded_scope_next88/source')/key;py='/root/miniconda3/envs/cybersim_agent/bin/python';env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu)
(out/'admission.json').write_text(json.dumps(dict(rule='both query-blind secondary descriptions classified A; parent retained without new certification',strict_all_four_A=[r['query_index'] for r in judgments if r['secondary_admitted']],admitted=ids,source_judgments=str(decisions),development_exploration=True,target_labels_used=False),indent=2)+'\n')
with (out/'execution.log').open('w') as log:
 if ids:
  subprocess.run([py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base','output/native_write_next75/all/calibrated','--cohort','all','--evidence-base','output/typed_evidence_next44','--output-base',str(out.parent),'--mode','seeded','--tracks-base',str(Path(tracks)/'union'),'--retained-selection',str(baseline/'typed_readout.pt'),'--extra-track-frames','--regional-positive-evidence','--scope-initialization','source','--query-indices',*ids],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 else:
  for name in ['field.pt','calibration.pt','typed_readout.pt']:shutil.copyfile(baseline/name,out/name)
 subprocess.run([py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(out.parent),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print(key,'complete',ids,flush=True)
