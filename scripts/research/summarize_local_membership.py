"""Complete paired mechanism comparison, with label diagnostics after prediction."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('output/local_membership_study'))
    a=p.parse_args();root=a.root
    controls={'v7':read(root/'v7_selection_agreement.json')}
    for name in ['transfer','scope']:
        complete=read(root/name/'predictions_complete.json')
        controls[name]=read(root/name/'uqis/identity_scope/result.json')
    metrics={}
    gains={}
    for name,report in controls.items():
        selection=report['explicit_selection'];paired=selection['paired_query_robustness']
        metrics[name]={m:dict(ap=v['scene_macro']['average_precision'],**selection['modalities'][m]['scene_macro']) for m,v in report['modalities'].items()}
        metrics[name]['paired']={k:v for k,v in paired.items() if k not in ['per_target','selection_agreement']}
        metrics[name]['agreement']={k:v for k,v in paired['selection_agreement'].items() if k not in ['per_target','interpretation']}
        if name=='v7':continue
        gains[name]={}
        for m in report['modalities']:
            gains[name][m]={}
            for measure in ['ap','iou']:
                values=report['modalities'][m] if measure=='ap' else selection['modalities'][m]
                old=controls['v7']['modalities'][m] if measure=='ap' else controls['v7']['explicit_selection']['modalities'][m]
                key='average_precision' if measure=='ap' else 'iou'
                delta={s:100*(v[key]-old['per_scene'][s][key]) for s,v in values['per_scene'].items()}
                gains[name][m][measure]=dict(scene_clustered_bootstrap(delta),per_scene=delta)
    public=read(root/'scope/summary.json');old=read(Path('output/system_v7/summary.json'))
    public_delta={b:{k:100*(v-old['macro'][b][k]) for k,v in public['macro'][b].items()} for b in ['lerf','nvos','scannet']}
    package=Path('output/scannet_uqis_9_v0_3_1_candidate');targets=read(package/'evaluator/targets.json')
    cases=[];builds=[]
    for domain in targets['scene_domains']:
        scene=domain['scene_id'];folder=root/'maps/uqis'/scene
        builds.append(read(folder/'build.json'))
        raw=sparse.load_npz(folder/'scope/membership.npz').tocsc()
        refined=sparse.load_npz(folder/'scope/local_membership.npz').tocsc()
        labels=np.load(package/'evaluator'/domain['mesh_instance_ids_path'])
        runs={arm:{v['query_id']:v for v in read(root/arm/'uqis'/scene/'query/registered/runs'/f'{scene}.json')['query_details']}
            for arm in ['transfer','scope']}
        for target in targets['targets']:
            if target['scene_id']!=scene:continue
            for m in ['text','image']:
                qid=target['queries'][m];seed=runs['scope'][qid]['instance_identity']['seed_region']
                before=raw[:,seed].toarray().reshape(-1)>.5;after=refined[:,seed].toarray().reshape(-1)>.5
                truth=labels==target['instance_id'];tp=int((before&truth).sum());fp=int((before&~truth).sum())
                cases.append(dict(scene_id=scene,query_id=qid,modality=m,seed_region=seed,
                    seed_unchanged=seed==runs['transfer'][qid]['instance_identity']['seed_region'],
                    original_true_members=tp,original_false_members=fp,
                    remaining_true_members=int((after&truth).sum()),remaining_false_members=int((after&~truth).sum())))
    diagnosis={m:{k:sum(v[k] for v in cases if v['modality']==m) for k in
        ['original_true_members','original_false_members','remaining_true_members','remaining_false_members']} for m in ['text','image']}
    output=dict(complete_predictions_before_each_scoring=complete['all_predictions_before_scoring'],
        uqis=metrics,gain_over_v7_pp=gains,public_macro=public['macro'],public_delta_pp=public_delta,
        seed_membership_diagnosis=diagnosis,seed_identity_unchanged=all(v['seed_unchanged'] for v in cases),
        build=dict(regions=sum(v['regions'] for v in builds),elements=sum(v['elements'] for v in builds),
            entries=sum(v['entries'] for v in builds),membership_below_half=sum(v['membership_below_half'] for v in builds),
            largest_view_mass_error=max(v['largest_view_mass_error'] for v in builds)),
        label_diagnosis_only=True,independent_test=False,cases=cases)
    (root/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k in ['public_delta_pp','seed_membership_diagnosis','build']},indent=2))
    for name,data in metrics.items():print(name,{m:{k:round(v*100,3) for k,v in values.items()} for m,values in data.items() if m not in ['paired','agreement']})


if __name__=='__main__':main()
