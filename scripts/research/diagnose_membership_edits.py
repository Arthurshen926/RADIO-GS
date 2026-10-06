"""Decompose a frozen candidate's actual additions/removals after prediction.

This is an evaluation-only explanation, never an inference gate or a source of
new labels. In particular, keeping a baseline member does not certify it as true.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=Path('output/system_v8_membership_closure_20261005'))
    parser.add_argument('--package',type=Path,default=Path('output/scannet_uqis_9_v0_3_1_candidate'))
    args=parser.parse_args()
    # The standard evaluator has already finished. No partial-query screening.
    scored=json.loads((args.system/'uqis/identity_scope/result.json').read_text())
    targets=json.loads((args.package/'evaluator/targets.json').read_text())
    labels={s['scene_id']:np.load(args.package/'evaluator'/s['mesh_instance_ids_path']) for s in targets['scene_domains']}
    rows=[]
    for target in targets['targets']:
        scene=target['scene_id'];truth=labels[scene]==target['instance_id']
        for modality in ['point_2d','point_3d']:
            qid=target['queries'][modality]
            base=np.load(args.reference/'uqis/identity_scope/selections'/f'{qid}.npy')
            new=np.load(args.system/'uqis/identity_scope/selections'/f'{qid}.npy')
            record=dict(scene=scene,modality=modality,query_id=qid,target_elements=int(truth.sum()))
            for name,mask in [('retained',base&new),('added',new&~base),('removed',base&~new),('missed_both',~base&~new)]:
                record[name]=dict(true=int((mask&truth).sum()),false=int((mask&~truth).sum()))
            rows.append(record)
    output=dict(reference=str(args.reference),candidate=str(args.system),evaluation_only=True,
                checkpoint_or_cutoff_changes=False,rows=rows,modalities={})
    for modality in ['point_2d','point_3d']:
        selected=[r for r in rows if r['modality']==modality];per_scene={}
        for scene in sorted({r['scene'] for r in selected}):
            group=[r for r in selected if r['scene']==scene]
            values={k:{label:sum(r[k][label] for r in group) for label in ['true','false']}
                    for k in ['retained','added','removed','missed_both']}
            rates=dict(added_true_fraction=values['added']['true']/max(sum(values['added'].values()),1),
                removed_true_fraction=values['removed']['true']/max(sum(values['removed'].values()),1),
                retained_true_fraction=values['retained']['true']/max(sum(values['retained'].values()),1),
                net_target_coverage=(values['added']['true']-values['removed']['true'])/sum(r['target_elements'] for r in group))
            per_scene[scene]=dict(element_counts=values,**rates)
        output['modalities'][modality]=dict(per_scene=per_scene,
            scene_macro={k:float(np.mean([r[k] for r in per_scene.values()])) for k in rates},
            actual_selection=scored['explicit_selection']['modalities'][modality]['scene_macro'])
    (args.system/'membership_edits.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({m:d['scene_macro'] for m,d in output['modalities'].items()},indent=2))


if __name__=='__main__':main()
