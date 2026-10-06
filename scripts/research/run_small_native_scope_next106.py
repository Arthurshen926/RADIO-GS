"""Matched old/expanded native-top1 source seed control for a predeclared zero query.
No verifier, aliases or GT-based candidate choice. Neither arm is a mainline change.
"""
import argparse,json,os,subprocess
from pathlib import Path
import torch
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--query-index',type=int,required=True);p.add_argument('--arm',choices=['old','expanded'],required=True);p.add_argument('--gpu',required=True);a=p.parse_args();key='lerf_'+a.scene;qi=a.query_index;load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
r=json.loads((Path('output/next106')/key/f'{qi}_unpaired.json').read_text());eligible=[x for x in r['rows'] if a.arm=='expanded' or not x['small']];winner=max(eligible,key=lambda x:x['native_score']);native=load(winner['path']);sam=load(native['sam_path']);masks=unpack_masks(sam['packed_masks'],sam['mask_shape'][1]);mask=masks[int(native['native_proposal_indices'][winner['local_index']])].copy();seed=dict(query=r['query'],query_index=qi,seed_frame=winner['frame'],seed_confidence=winner['native_score'],masks=torch.from_numpy(mask)[None].bool(),source='native top1 before pair/VLM admission; bounded source-only control',candidate=winner,arm=a.arm)
seedbase=Path('output/small_native_seeds_next106')/a.arm;out=seedbase/key;out.mkdir(parents=True,exist_ok=True);torch.save(seed,out/f'{qi}.pt');(out/f'{qi}.json').write_text(json.dumps(winner,indent=2)+'\n')
trackbase=Path('output/small_native_tracks_next106')/a.arm;scopebase=Path('output/small_native_scope_next106')/a.arm;out=scopebase/key;out.mkdir(parents=True,exist_ok=True);parent=Path('output/core_scope_next104/core_update')/key/'typed_readout.pt';env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=a.gpu);py='/root/miniconda3/envs/cybersim_agent/bin/python'
with (out/'execution.log').open('w') as log:
 subprocess.run(['/root/miniconda3/envs/sam3/bin/python','scripts/research/track_full_text_sources_next79.py','--key',key,'--query-indices',str(qi),'--seed-base',str(seedbase),'--output-base',str(trackbase),'--raster-scale','2'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([py,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--semantic-base','output/native_write_next75/all/calibrated','--cohort','all','--evidence-base','output/typed_evidence_next44','--output-base',str(scopebase),'--mode','seeded','--tracks-base',str(trackbase),'--retained-selection',str(parent),'--extra-track-frames','--regional-positive-evidence','--scope-initialization','source','--negative-admission','corroborated','--raster-scale','2','--pixel-loss','balanced','--support-prior','footprint','--support-prior-weight','0.1','--query-indices',str(qi)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([py,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(scopebase),'--readout','anchored','--pipeline','--calibrated','--element-results',str(out/'typed_readout.pt')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print(key,a.arm,'complete',flush=True)
