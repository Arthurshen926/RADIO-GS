"""Source-only diagnostic: best possible scalar threshold on native union scores.

Each atom's union admission score is the maximum covering proposal score.
Thresholds only produce tied-score prefixes. This oracle uses source pseudo
masks per query/view; it is never a deployable policy or a target-GT bound.
"""
import gc
from pathlib import Path
import numpy as np
import torch
from radio_gs.query_field.storage import sha256
from scripts.research.run_query_field_scope_topology import ROOT as PARENT, PARENT as REFERENCE
from scripts.research.run_query_field_signed_scope import ROOT
from scripts.research.run_query_field_relations import read,write,receipt


def threshold_capacity(scores, area, positive, covered):
    total=float(positive.sum())
    if total==0:return None
    ids=np.flatnonzero(covered & (area>0))
    if not len(ids):return dict(iou=0.,selected_atoms=0,score_minimum=None,excluded_maximum=None)
    order=ids[np.argsort(-scores[ids],kind='stable')]
    tp=np.cumsum(positive[order]);mass=np.cumsum(area[order])
    # A threshold cannot split ties even when their target purities differ.
    ends=np.r_[scores[order[:-1]]!=scores[order[1:]],True]
    options=np.flatnonzero(ends);iou=tp/(total+mass-tp)
    best=int(options[np.argmax(iou[options])])
    return dict(iou=float(iou[best]),selected_atoms=best+1,score_minimum=float(scores[order[best]]),
                excluded_maximum=float(scores[order[best+1]]) if best+1<len(order) else None)


def main():
    cfg=read(ROOT/'prediction_config.json');scenes={}
    for job in cfg['jobs']:
        scene=job['scene'];root=Path(job['field']);manifest=read(root/'manifest.json')
        assert sha256(root/'field.pt')==manifest['files']['field.pt']['sha256']
        state=torch.load(root/'field.pt',weights_only=True)
        prediction=torch.load(PARENT/'predictions'/job['name']/'predictions.pt',weights_only=True)
        fixed=torch.load(REFERENCE/'predictions'/job['name']/'privileged_scope.pt',weights_only=True)
        previous=read(PARENT/'predictions'/job['name']/'prediction.json')
        assert prediction['texts']==fixed['texts']
        q=prediction['native_query_vectors'].double().numpy();q/=np.linalg.norm(q,axis=1,keepdims=True)
        n=prediction['generic_negative_vectors'].double().numpy();n/=np.linalg.norm(n,axis=1,keepdims=True)
        rows=[]
        for vi in range(len(fixed['views'])):
            prefix=f'scope_topology.views.{vi}.'
            a=state[prefix+'incidence'].numpy();center=state[prefix+'neighborhood'][:,4].numpy()
            z=state[prefix+'semantic'].double().numpy();z/=np.linalg.norm(z,axis=1,keepdims=True)
            p=1/(1+np.exp(-10*(z@q.T-(z@n.T).max(1)[:,None])))
            valid=state[f'extent_preparation.views.{vi}.valid'].numpy()
            area=np.bincount(center,weights=valid,minlength=len(a));covered=a.any(1)
            for qi,text in enumerate(fixed['texts']):
                truth=fixed['views'][vi]['target'][:,qi].numpy()>.5
                positive=np.bincount(center,weights=truth&valid,minlength=len(a))
                upper=np.where(a,p[:,qi][None],0.).max(1)
                result=threshold_capacity(upper,area,positive,covered)
                if result is None:continue
                native=covered&(upper>.5)
                tp=float(positive[native].sum());native_iou=tp/max(float(area[native].sum()+positive.sum()-tp),1)
                old=previous['diagnostics'][qi]['capacity'][vi]
                assert abs(native_iou-old['native_iou'])<1e-12
                assert result['iou']+1e-12>=native_iou
                rows.append(dict(query=text,view=vi,**result,native_iou=native_iou,
                                 proposal_greedy_iou=old['full']['greedy_iou']))
        scenes[scene]=dict(rows=rows,count=len(rows),threshold_oracle_iou=float(np.mean([r['iou'] for r in rows])),
            native_iou=float(np.mean([r['native_iou'] for r in rows])),proposal_greedy_iou=float(np.mean([r['proposal_greedy_iou'] for r in rows])),
            field_manifest=receipt(root/'manifest.json'),field_sha256=manifest['files']['field.pt']['sha256'])
        print(scene,{k:v for k,v in scenes[scene].items() if k not in ['rows','field_manifest','field_sha256']},flush=True)
        del state,prediction,fixed;gc.collect()
    result=dict(scenes=scenes,posthoc=True,source_pseudo_targets=True,target_ground_truth_used=False,
        per_query_view_oracle=True,deployable_threshold=False,global_gaussian_performance_upper_bound=False,
        code=receipt(__file__),prediction_design=receipt(ROOT/'design.json'))
    write(ROOT/'native_order_capacity.json',result);write('docs/experiments/2026-10-12-native-order-capacity.json',result)


if __name__=='__main__':torch.set_num_threads(4);main()
