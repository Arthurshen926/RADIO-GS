"""After scoring, locate missing target support and actual set edits; no inference."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def read(path):return json.loads(path.read_text())


def main():
    root=Path('output/surface_membership_study')
    system=root/'learned_system'
    parent=Path('output/learned_instance_additions/element_query_system')
    package=Path('output/scannet_uqis_9_v0_3_1_candidate')
    # Run only after the complete frozen cohort has been scored.
    scored=read(system/'uqis/identity_scope/result.json')
    targets=read(package/'evaluator/targets.json')
    rows=[]
    for scene in targets['scene_domains']:
        name=scene['scene_id'];folder=root/'maps/learned'/name
        labels=np.load(package/'evaluator'/scene['mesh_instance_ids_path'])
        membership=sparse.load_npz(folder/'membership.npz').tocsc()
        visibility=sparse.load_npz(folder/'visibility.npz').tocsc()
        frames=np.repeat(np.arange(visibility.shape[1]),np.diff(np.load(folder/'region_offsets.npy')))
        details={r['query_id']:r for r in read(system/'uqis'/name/'query/registered/runs'/f'{name}.json')['query_details']}
        observable=membership.getnnz(axis=1)>0
        for target in [r for r in targets['targets'] if r['scene_id']==name]:
            truth=labels==target['instance_id'];size=int(truth.sum())
            for modality,qid in target['queries'].items():
                d=details[qid]
                if modality.startswith('point'):
                    member=d['conditional_membership']
                    if member['applied']:
                        seed=member['seed_region']
                        shared=np.asarray((membership.T@membership[:,seed]).toarray()).ravel()
                        regions=np.flatnonzero(shared>0)
                        domain=np.asarray(membership[:,regions].sum(1)).ravel()>0
                    else:domain=np.zeros(len(labels),bool)
                    positive=domain
                else:
                    regions=d['instance_identity']['matched_regions']
                    domain=np.asarray(visibility[:,np.unique(frames[regions])].sum(1)).ravel()>0
                    positive=np.asarray(membership[:,regions].sum(1)).ravel()>0
                old=np.load(parent/'uqis/identity_scope/selections'/f'{qid}.npy')
                new=np.load(system/'uqis/identity_scope/selections'/f'{qid}.npy')
                added=new&~old;removed=old&~new
                rows.append(dict(scene=name,modality=modality,query=qid,
                    source_observable_fraction=float((truth&observable).sum()/size),
                    candidate_domain_fraction=float((truth&domain).sum()/size),
                    positive_region_union_fraction=float((truth&positive).sum()/size),
                    parent_target_coverage=float((truth&old).sum()/size),
                    final_target_coverage=float((truth&new).sum()/size),
                    added_true=int((added&truth).sum()),added_false=int((added&~truth).sum()),
                    removed_true=int((removed&truth).sum()),removed_false=int((removed&~truth).sum())))
    output=dict(evaluation_only=True,labels_do_not_select_inference_branch=True,
        note='Support fractions explain missing evidence; they are not achievable method scores or an oracle decoder.',
        rows=rows,modalities={})
    for modality in ['text','image','point_2d','point_3d']:
        selected=[r for r in rows if r['modality']==modality];scenes={}
        for scene in sorted({r['scene'] for r in selected}):
            group=[r for r in selected if r['scene']==scene]
            counts={k:sum(r[k] for r in group) for k in ['added_true','added_false','removed_true','removed_false']}
            scenes[scene]={k:float(np.mean([r[k] for r in group])) for k in selected[0]
                if k.endswith('_fraction') or k.endswith('_coverage')}
            scenes[scene].update(added_true_fraction=counts['added_true']/max(counts['added_true']+counts['added_false'],1),
                removed_false_fraction=counts['removed_false']/max(counts['removed_true']+counts['removed_false'],1),
                **counts)
        output['modalities'][modality]=dict(per_scene=scenes,
            scene_macro={k:float(np.mean([s[k] for s in scenes.values()])) for k in next(iter(scenes.values()))},
            actual_set=scored['explicit_selection']['modalities'][modality]['scene_macro'])
    (root/'selection_diagnosis.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({m:r['scene_macro'] for m,r in output['modalities'].items()},indent=2))


if __name__=='__main__':main()
