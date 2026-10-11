"""Exact source-pseudo-mask IoU capacity of disjoint topology atoms.

Posthoc diagnosis, not a deployable selector or target-GT performance bound.
For disjoint atoms, an IoU optimum is a prefix ordered by positive fraction:
adding an atom improves IoU iff t/(a-t) exceeds the current IoU. Thus every
optimum can be represented by a threshold on t/a (ties do not break capacity).
"""
import gc
from pathlib import Path
import numpy as np
import torch
from radio_gs.query_field.storage import load_field
from scripts.research.run_query_field_scope_topology import ROOT,PARENT
from scripts.research.run_query_field_relations import read,write,receipt


def atom_capacity(center,valid,target,atoms):
    area=np.bincount(center,weights=valid.astype(np.float64),minlength=atoms)
    positive=np.bincount(center,weights=(valid&target).astype(np.float64),minlength=atoms)
    total=positive.sum()
    if total==0:return None
    ids=np.flatnonzero(area>0);fraction=positive[ids]/area[ids]
    order=ids[np.argsort(-fraction,kind='stable')]
    tp=np.cumsum(positive[order]);size=np.cumsum(area[order]);iou=tp/(total+size-tp)
    best=int(iou.argmax())
    return dict(iou=float(iou[best]),selected_atoms=best+1,observed_atoms=len(ids),positive_pixels=int(total),
                split_impure_atoms=int(((positive>0)&(positive<area)).sum()))


def main():
    cfg=read(ROOT/'prediction_config.json');scenes={}
    for j in cfg['jobs']:
        scene=j['scene'];field,manifest=load_field(ROOT/'fields'/scene/'field')
        fixed=torch.load(PARENT/'predictions'/j['name']/'privileged_scope.pt',weights_only=True)
        predicted=read(ROOT/'predictions'/j['name']/'prediction.json');rows=[]
        for vi,(v,pv) in enumerate(zip(field.scope_topology.views,field.extent_preparation.views)):
            center=v.neighborhood[:,4].numpy();valid=pv.valid.numpy()
            for qi,text in enumerate(fixed['texts']):
                cap=atom_capacity(center,valid,fixed['views'][vi]['target'][:,qi].numpy()>.5,len(v.incidence))
                if cap is None:continue
                old=predicted['diagnostics'][qi]['capacity'][vi]
                assert cap['iou']+1e-6>=old['full']['greedy_iou']
                rows.append(dict(query=text,view=vi,**cap,proposal_greedy_iou=old['full']['greedy_iou'],
                                 retained_greedy_iou=old['retained']['greedy_iou'],native_iou=old['native_iou'],
                                 greedy_saturated=old['full']['selected']==16))
        scenes[scene]=dict(rows=rows,count=len(rows),atom_iou=float(np.mean([r['iou'] for r in rows])),
            proposal_iou=float(np.mean([r['proposal_greedy_iou'] for r in rows])),
            retained_iou=float(np.mean([r['retained_greedy_iou'] for r in rows])),
            native_iou=float(np.mean([r['native_iou'] for r in rows])),greedy_saturated=sum(r['greedy_saturated'] for r in rows),
            field=receipt(ROOT/'fields'/scene/'field/manifest.json'))
        print(scene,{k:v for k,v in scenes[scene].items() if k not in ['rows','field']},flush=True)
        del field,fixed;gc.collect()
    result=dict(scenes=scenes,posthoc=True,privileged_source_masks=True,target_ground_truth_used=False,
        exact_for='unrestricted unions of existing atoms, fixed valid construction rays and binary source pseudo target only',
        global_gaussian_performance_upper_bound=False,stored_query_answers=False,
        code=receipt(__file__),parent_prediction_design=receipt(ROOT/'design.json'))
    write(ROOT/'atom_capacity.json',result);write('docs/experiments/2026-10-11-scope-atom-capacity.json',result)


if __name__=='__main__':torch.set_num_threads(4);main()
