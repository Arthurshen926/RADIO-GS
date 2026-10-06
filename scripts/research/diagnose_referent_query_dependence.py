"""Post-fit intervention: preserve native scores, change only learned query correction.

Uses independent labelled rooms for diagnosis, never benchmark prediction. Query
mean and wrong-target interventions are deliberately not deployment methods.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from radio_gs.querying.scene_referent_grounding import ReferentGrounder
from scripts.research.fit_referent_grounder import read_scene

DATA=Path('output/real_referent_grounding_20261005')


@torch.inference_mode()
def diagnose(checkpoint,output,splits):
    torch.set_num_threads(3);device='cuda'
    saved=torch.load(checkpoint,map_location=device,weights_only=False)
    model=ReferentGrounder(**saved['architecture']).to(device).eval();model.load_state_dict(saved['state_dict'])
    design=json.loads((DATA/'design.json').read_text());report={}
    for split in splits:
        records=[]
        for scene in design[split]:
            data=read_scene(scene,device,DATA);native=data['pooled']@data['visual'].T*model.native_scale
            scores=[]
            for i in range(len(data['rows'])):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    scores.append(model(data['visual'],data['geometry'],data['tokens'][i:i+1],data['valid'][i:i+1],data['pooled'][i:i+1])[0])
            learned=torch.stack(scores);delta=learned-native
            # Per-query additive constants do not change a candidate ranking.
            centred=delta-delta.mean(1,keepdim=True);common=centred.mean(0,keepdim=True)
            identity=np.array([q['raw_object_id'] for q in data['rows']]);permuted=[]
            for i in range(len(identity)):
                alternatives=np.flatnonzero(identity!=identity[i]);after=alternatives[alternatives>i]
                permuted.append(int(after[0] if len(after) else alternatives[0]))
            conditions=dict(native=native,original=learned,query_mean=native+common,wrong_target_query=native+delta[permuted])
            row=torch.arange(len(identity),device=device);measured={}
            for name,values in conditions.items():
                seeds=values.argmax(1)
                measured[name]=dict(iou=float(data['reward'][row,seeds].mean()),
                    identity_accuracy=float((data['purity'][row,seeds]>.5).float().mean()),
                    same_seed_as_original=float((seeds==learned.argmax(1)).float().mean()),
                    per_query_iou=data['reward'][row,seeds].cpu().tolist())
            variance=float(centred.square().mean());common_variance=float(common.square().mean())
            total_delta=delta-delta.mean()
            query_offset=delta.mean(1,keepdim=True)-delta.mean()
            interaction=centred-common
            total_energy=max(float(total_delta.square().mean()),1e-12)
            records.append(dict(scene=scene,queries=len(identity),common_correction_energy_fraction=common_variance/max(variance,1e-12),
                correction_energy_decomposition=dict(query_offset=float(query_offset.square().mean())/total_energy,
                    regional_prior=common_variance/total_energy,query_region_interaction=float(interaction.square().mean())/total_energy),
                conditions=measured))
            print(split,scene,{k:round(v['iou']*100,3) for k,v in measured.items()},flush=True)
            del data,native,learned,delta,scores,centred,common,conditions;torch.cuda.empty_cache()
        report[split]=dict(scenes=records,common_correction_energy_fraction=float(np.mean([r['common_correction_energy_fraction'] for r in records])),
            correction_energy_decomposition={k:float(np.mean([r['correction_energy_decomposition'][k] for r in records]))
                for k in ['query_offset','regional_prior','query_region_interaction']},
            conditions={k:{metric:float(np.mean([r['conditions'][k][metric] for r in records])) for metric in ['iou','identity_accuracy','same_seed_as_original']}
                for k in ['native','original','query_mean','wrong_target_query']})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(checkpoint=str(checkpoint),diagnostic_only=True,benchmark_queries_used=False,groups=report),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--splits',nargs='+',default=['calibration','validation']);a=p.parse_args();diagnose(a.checkpoint,a.output,a.splits)
