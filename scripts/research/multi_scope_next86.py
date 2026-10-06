"""Bounded text-union hypotheses; source-only choice and matched unknown handling."""
import argparse,json,os,subprocess
from pathlib import Path
from functools import lru_cache
import torch
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks

def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def propose(scene, seed_base="output/grounded_seeds_next85", output_base="output/multi_seeds_next86"):
 key='lerf_'+scene;relation=load(Path('output/region_relations_next79')/scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text());nodes=[]
 for e in manifest['entries']:
  nodes.extend((e['frame'],i,e['path']) for i in range(e['regions']))
 @lru_cache(maxsize=8)
 def bank(path):return load(path)
 def support(node):
  _,ri,path=nodes[node];r=bank(path);take=(r['columns']==ri)&(r['values']>.5);return set(r['ids'][r['rows'][take].long()].tolist())
 out=Path(output_base)/key;out.mkdir(parents=True,exist_ok=True);rows=json.loads((Path(seed_base)/key/'proposals.json').read_text());report=[]
 for row in rows:
  primary=row['selected_rank'];qi=row['query_index'];secondary=None;overlaps={}
  if primary is not None:
   candidates=row['candidates'];base=set().union(*(support(n) for n in candidates[primary]['nodes']));margins=torch.tensor(row['candidate_margin']).amin(1)
   for rank in torch.argsort(margins,descending=True).tolist():
    if rank==primary or margins[rank]<=0:continue
    sig=set().union(*(support(n) for n in candidates[rank]['nodes']));overlap=len(base&sig)/max(1,min(len(base),len(sig)));overlaps[rank]=overlap
    # Symmetric intersection/min rejects both same-extent and containment pairs.
    # A bounded two-hypothesis pilot, not exhaustive instance discovery.
    if base and sig and overlap<=.2:
     secondary=rank;break
  record=dict(query=row['query'],query_index=qi,primary_rank=primary,secondary_rank=secondary,overlap_with_primary=overlaps)
  if secondary is not None:
   c=row['candidates'][secondary];side=int(torch.tensor(row['candidate_margin'][secondary]).argmax());node=c['nodes'][side];frame,ri,path=nodes[node];r=bank(path);native=load(r['source']);sam=load(native['sam_path']);masks=unpack_masks(sam['packed_masks'],sam['mask_shape'][1]);mask=masks[int(native['native_proposal_indices'][ri])].copy()
   seed=dict(query=row['query'],query_index=qi,seed_frame=frame,seed_confidence=max(0.,c['native_score']),masks=torch.from_numpy(mask)[None].bool(),pair_nodes=c['nodes'],pair_frames=c['frames'],source='second spatially distinct two-view accepted text hypothesis',single_scope_hypothesis=True)
   torch.save(seed,out/f'{qi}.pt');record['seed_frame']=frame
  report.append(record)
 (out/'proposals.json').write_text(json.dumps(report,indent=2)+'\n');return report

def merge(key,rows, primary_base="output/grounded_tracks_next85", secondary_base="output/secondary_tracks_next86", output_base="output/multi_tracks_next86"):
 for policy in ['union','unknown']:
  out=Path(output_base)/policy/key;out.mkdir(parents=True,exist_ok=True)
  for row in rows:
   qi=row['query_index'];primary=load(Path(primary_base)/key/f'{qi}.pt')
   if row['secondary_rank'] is None:torch.save(primary,out/f'{qi}.pt');continue
   secondary=load(Path(secondary_base)/key/f'{qi}.pt');frames={}
   for frame in primary['frames']:
    records=[t['frames'].get(frame,{}) for t in [primary,secondary]];present=[bool(r.get('present',False)) and frame!=t['seed_frame'] for r,t in zip(records,[primary,secondary])];shape=primary['frames'][frame]['probability'].shape
    probabilities=[r['probability'].float() if active else torch.zeros(shape) for r,active in zip(records,present)];probability=torch.stack(probabilities).amax(0)
    r=dict(probability=probability.half(),present=any(present),hypotheses_present=present)
    if policy=='unknown':r.update(positive_allowed=torch.ones(shape,dtype=torch.bool),negative_allowed=torch.full(shape,all(present),dtype=torch.bool),scope_reliability=1.)
    frames[frame]=r
   seed=dict(primary['seed_evidence'],single_scope_hypothesis=False,query_type='text_union',hypotheses=[primary['seed_evidence'],secondary['seed_evidence']],confidence_rule='retain primary confidence for matched union policies')
   torch.save(dict(query=primary['query'],frames=frames,seed_frame=None,seed_evidence=seed,merge_policy=policy,source_tracks=[str(Path(primary_base)/key/f'{qi}.pt'),str(Path(secondary_base)/key/f'{qi}.pt')]),out/f'{qi}.pt')

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True);p.add_argument('--seed-base',default='output/grounded_seeds_next85');p.add_argument('--primary-tracks',default='output/grounded_tracks_next85');p.add_argument('--baseline-base',default='output/grounded_scope_next85/source');p.add_argument('--iteration',default='next86');p.add_argument('--admission-base');p.add_argument('--policies',nargs='+',default=['union','unknown']);a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene;seed_out='output/multi_seeds_'+a.iteration;secondary_out='output/secondary_tracks_'+a.iteration;merged_out='output/multi_tracks_'+a.iteration;rows=propose(a.scene,a.seed_base,seed_out)
 if a.admission_base:
  decision=Path(a.admission_base)/key/'joint_identity.json'
  admitted={r['query_index'] for r in json.loads(decision.read_text())['results'] if r['secondary_admitted']} if decision.exists() else set()
  for row in rows:
   if row['query_index'] not in admitted:row['rejected_secondary_rank']=row['secondary_rank'];row['secondary_rank']=None
  (Path(seed_out)/key/'proposals.json').write_text(json.dumps(rows,indent=2)+'\n')
 ids=[str(r['query_index']) for r in rows];extra=[str(r['query_index']) for r in rows if r['secondary_rank'] is not None];root=Path('output/multi_scope_'+a.iteration);root.mkdir(exist_ok=True);env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);py='/root/miniconda3/envs/cybersim_agent/bin/python'
 print(key,'secondary queries',extra,flush=True)
 if extra:
  with (root/(key+'_tracking.log')).open('w') as log:subprocess.run(['/root/miniconda3/envs/sam3/bin/python','scripts/research/track_full_text_sources_next79.py','--key',key,'--query-indices',*extra,'--seed-base',seed_out,'--output-base',secondary_out],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 merge(key,rows,a.primary_tracks,secondary_out,merged_out)
 for policy in a.policies:
  out=root/policy/key;out.mkdir(parents=True,exist_ok=True)
  with (out/'execution.log').open('w') as log:
   subprocess.run([py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base','output/native_write_next75/all/calibrated','--cohort','all','--evidence-base','output/typed_evidence_next44','--output-base',str(out.parent),'--mode','seeded','--tracks-base',str(Path(merged_out)/policy),'--retained-selection',str(Path(a.baseline_base)/key/'typed_readout.pt'),'--extra-track-frames','--regional-positive-evidence','--scope-initialization','source','--query-indices',*ids],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   subprocess.run([py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(out.parent),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  print(key,policy,'complete',flush=True)
if __name__=='__main__':main()
