"""Post-score attribution of input registration, identity and member scope.

Labels describe saved outputs only. No anchor, model or threshold is selected.
The dominant support is a diagnostic for mixtures, not their actual condition.
"""
from pathlib import Path

import numpy as np
import torch

from scripts.research.run_query_field_joint_observation import ROOT, PREVIOUS, ARMS, read, write, receipt
from scripts.research.run_query_field_surface_input import ROOT as SURFACE


def main():
    read(ROOT/'scoring_complete.json'); read(SURFACE/'summary.json')
    design=read(ROOT/'design.json')
    package=Path(read(design['arms']['full_supervised']['path'])['uqis_package'])
    targets=read(package/'evaluator/targets.json'); records=[]; evidence=[]
    paths={a:ROOT/a/'predictions/uqis' for a in ARMS}
    paths['prior']=PREVIOUS/'confidence_mixture_function/predictions/uqis'
    paths['surface']=SURFACE
    for job in design['fields']:
        name=job['scene']; domain=next(d for d in targets['scene_domains'] if d['scene_id']==name)
        source=read(Path(job['observations'])/'observations.json')
        geometry=torch.load(source['geometry']['path'],map_location='cpu',weights_only=True)
        mesh=np.load(package/'method'/domain['mesh_xyz_path'])
        labels=np.load(package/'evaluator'/domain['mesh_instance_ids_path'])
        ids=geometry['source_vertex_ids'].numpy(); row_labels=labels[ids]
        if not np.array_equal(geometry['state_dict']['xyz'].numpy(),mesh[ids]):
            raise ValueError('carrier mapping changed')
        for target in [t for t in targets['targets'] if t['scene_id']==name]:
            truth=labels==target['instance_id']
            for modality,qid in target['queries'].items():
                for arm,path in paths.items():
                    if arm=='surface' and modality!='point_2d':continue
                    saved=torch.load(path/'gaussians'/(qid+'.pt'),map_location='cpu',weights_only=True)
                    candidates=saved['candidates'].numpy(); weights=saved['candidate_weights'].numpy()
                    positive=weights>0; candidates=candidates[positive]; weights=weights[positive]
                    correct=row_labels[candidates]==target['instance_id']
                    weights=weights/weights.sum() if len(weights) else weights
                    selected=np.load(path/'selections'/(qid+'.npy'))
                    tp=int((selected&truth).sum()); fp=int((selected&~truth).sum()); fn=int((~selected&truth).sum())
                    records.append(dict(scene=name,query_id=qid,modality=modality,arm=arm,
                        candidates=len(candidates),unsupported=not len(candidates),
                        dominant_correct=bool(len(correct) and correct[weights.argmax()]),
                        any_correct=bool(correct.any()),all_correct=bool(len(correct) and correct.all()),
                        correct_weight=float(weights[correct].sum()),
                        precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),iou=tp/max(tp+fp+fn,1)))
        evidence.append(receipt(package/'evaluator'/domain['mesh_instance_ids_path']))
        print('input attribution',name,flush=True)
    summary={}
    for arm in paths:
        summary[arm]={}
        for modality in ['text','image','point_2d','point_3d']:
            rows=[r for r in records if r['arm']==arm and r['modality']==modality]
            if arm=='surface' and modality!='point_2d':continue
            if len(rows)!=67:raise ValueError('incomplete attribution cohort')
            subsets={}
            for group,subset in [('all',rows),('dominant_correct',[r for r in rows if r['dominant_correct']]),
                                 ('dominant_incorrect',[r for r in rows if not r['dominant_correct']])]:
                subsets[group]=dict(queries=len(subset),**{k:float(np.mean([r[k] for r in subset])) if subset else None
                    for k in ['precision','recall','iou']})
            summary[arm][modality]=dict(queries=len(rows),**{k:sum(r[k] for r in rows)
                for k in ['dominant_correct','any_correct','all_correct','unsupported']},
                mean_correct_weight=float(np.mean([r['correct_weight'] for r in rows])),subsets=subsets)
    write(ROOT/'diagnostics/input_identity.json',dict(summary=summary,records=records,
        labels_used='post-score description only',model_change=False,oracle_selection=False,
        limitations=['query means, not benchmark scene macros',
            '2D mixture dominant support is diagnostic; all weighted supports condition its member function',
            'text/image softmax weights describe alternatives, not calibrated target probabilities',
            'conditioning on correct identity is descriptive and is not a causal comparison'],
        code=receipt(Path(__file__)),evidence=evidence+[receipt(ROOT/'predictions_complete.json'),
            receipt(SURFACE/'predictions_complete.json')]))


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
