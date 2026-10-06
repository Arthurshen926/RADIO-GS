"""One fixed study: observation repair, identity vs extent, full task replay."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap

ROOT=Path('output/real_referent_grounding_20261005')
CONDITIONS=['local_scope','scene_scope','scene_identity']


def read(path):return json.loads(path.read_text())


def capacity():
    design=read(ROOT/'design.json');groups={}
    for split in ['fit','calibration','validation']:
        rows=[]
        for scene in design[split]:
            folder=ROOT/'maps'/scene;spec=read(folder/'map.json')
            target=np.load(folder/'targets.npz');membership=sparse.load_npz(folder/'membership.npz')
            labels=np.load(ROOT/'labels'/(scene+'.npy'));support=membership.getnnz(axis=1)>0
            area=np.asarray(membership.sum(0)).ravel()
            purity=target['region_intersection']/np.maximum(area[:,None],1)
            rows.append(dict(scene=scene,frames=spec['source_count'],regions=spec['regions'],map_bytes=spec['map_bytes'],
                targets=len(target['instance_ids']),majority_targets=int(np.count_nonzero(purity.max(0)>.5)),
                coverage=float(np.mean([support[labels==i].mean() for i in target['instance_ids']])),
                finite_scope_oracle=float(target['scope_iou'].max(0).mean())))
        groups[split]=dict(rooms=len(rows),frames=sum(r['frames'] for r in rows),regions=sum(r['regions'] for r in rows),
            targets=sum(r['targets'] for r in rows),majority_targets=sum(r['majority_targets'] for r in rows),
            scene_mean_coverage=float(np.mean([r['coverage'] for r in rows])),
            scene_mean_finite_scope_oracle=float(np.mean([r['finite_scope_oracle'] for r in rows])),scenes=rows)
    (ROOT/'capacity.json').write_text(json.dumps(groups,indent=2)+'\n')
    return groups


def independent_quality():
    """Measure identity purity and complete-set coverage separately, after fitting."""
    result={}
    for condition in CONDITIONS:
        fit=read(ROOT/condition/'fit.json');rows=[]
        for record in fit['validation']['scenes']:
            scene=record['scene'];folder=ROOT/'maps'/scene
            selections=sparse.load_npz(folder/'scope_selections.npz').tocsc()
            labels=np.load(ROOT/'labels'/(scene+'.npy'))
            queries=read(ROOT/'language'/(scene+'.json'));values=[]
            for index,query in enumerate(queries):
                selected=selections[:,record['per_query_seed'][index]].indices
                truth=labels==query['raw_object_id']+1;intersection=truth[selected].sum()
                values.append(dict(seed_purity=record['per_query_purity'][index],
                    set_purity=float(intersection/max(len(selected),1)),coverage=float(intersection/truth.sum()),
                    selected_to_target_size=float(len(selected)/truth.sum())))
            rows.append(dict(scene=scene,**{key:float(np.mean([v[key] for v in values])) for key in values[0]}))
        result[condition]=dict(scene_macro={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key!='scene'},scenes=rows)
    (ROOT/'independent_selection_quality.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main(full_condition='scene_identity',declaration='configs/recipes/real_referent_grounding.json',pairs=()):
    baseline=Path('output/identity_bottleneck_20261005/baseline_system')
    systems={'native':baseline,**{name:ROOT/name/'system' for name in CONDITIONS}}
    results={name:read(folder/'uqis/identity_scope/result.json') for name,folder in systems.items()}
    dataset_root=Path('output/scannet_uqis_9_v0_3_1_candidate/evaluator')
    dataset=read(dataset_root/'targets.json')
    base_maps={j['name']:Path(j['scene']).parent for j in read(Path('configs/pipeline/system_v8.json'))['jobs'] if j['benchmark']=='uqis'}
    report=dict(conditions={},fits={},training_capacity=capacity(),independent_selection_quality=independent_quality(),
        default_changed=False,independent_final_test=False,
        declaration=declaration,extra_supervision='independent NR3D and official3D instance labels',
        fresh_jobs=sum(read(ROOT/name/'system/summary.json')['completed'] for name in CONDITIONS),
        public_task_replay=read(ROOT/full_condition/'system/summary.json')['macro'])
    for condition in CONDITIONS:
        fit=read(ROOT/condition/'fit.json')
        report['fits'][condition]={k:fit[k] for k in ['condition','supervision_target','microbatch','epochs','selected_epoch',
            'training_listener_correct_records','excluded_training_zero_scope_support','seconds','parameter_count']}
        report['fits'][condition].update({k:fit[k] for k in ['objective','eligibility','training_eligible_unique_records',
            'training_query_presentations_per_epoch','include_native_checkpoint','batch_organization'] if k in fit})
        for split in ['calibration','validation']:
            data=fit[split]
            report['fits'][condition][split]={k:v for k,v in data.items() if k!='scenes'}
            report['fits'][condition][split]['per_scene']=[{k:v for k,v in row.items() if not k.startswith('per_query')} for row in data['scenes']]
            report['fits'][condition][split]['iou_delta_pp']=scene_clustered_bootstrap({
                row['scene']:100*(row['iou']-row['native_iou']) for row in data['scenes']})
        train=read(ROOT/condition/'training_fit_diagnosis.json')
        report['fits'][condition]['training_fit']={k:v for k,v in train.items() if k!='scenes'}
    for condition,result in results.items():
        system=systems[condition];metrics={}
        for modality in ['text','image','point_2d','point_3d']:
            selected=result['explicit_selection']['modalities'][modality]
            original=results['native']['explicit_selection']['modalities'][modality]
            metrics[modality]=dict(ap=result['modalities'][modality]['scene_macro']['average_precision'],
                **selected['scene_macro'],iou_delta_pp=scene_clustered_bootstrap({scene:100*(row['iou']-
                    original['per_scene'][scene]['iou']) for scene,row in selected['per_scene'].items()}))
        cases=[]
        for domain in dataset['scene_domains']:
            scene=domain['scene_id'];labels=np.load(dataset_root/domain['mesh_instance_ids_path'])
            membership=sparse.load_npz(base_maps[scene]/'membership.npz').tocsc()
            area=np.asarray(membership.sum(0)).ravel();available=np.flatnonzero(area>0)
            evidence=np.load(system/'uqis'/scene/'query/registered/identity_evidence.npz')
            evidence_columns={q:i for i,q in enumerate(evidence['query_ids'])}
            rows=read(system/'uqis'/scene/'query/registered/runs'/f'{scene}.json')['query_details']
            details={row['query_id']:row for row in rows}
            for target in dataset['targets']:
                if target['scene_id']!=scene:continue
                qid=target['queries']['text'];seed=details[qid]['instance_identity']['seed_region']
                vertices=membership[:,seed].indices;purity=float(np.mean(labels[vertices]==target['instance_id']))
                ids,counts=np.unique(labels[vertices],return_counts=True);dominant=int(ids[counts.argmax()])
                kind=('correct' if dominant==target['instance_id'] else 'same_class_distractor'
                    if dominant in target['same_class_distractor_instance_ids'] else 'other_or_unlabelled')
                target_fraction=np.asarray(membership.T@(labels==target['instance_id']).astype(np.float32)).ravel()/np.maximum(area,1)
                order=available[np.argsort(-evidence['identity_response'][available,evidence_columns[qid]],kind='stable')]
                correct=np.flatnonzero(target_fraction[order]>.5)
                cases.append(dict(scene_id=scene,query_id=qid,seed=seed,seed_purity=purity,identity_error_type=kind,
                    first_majority_candidate_rank=int(correct[0]+1) if len(correct) else None))
        unchanged={}
        for modality in ['image','point_2d','point_3d']:
            queries=[target['queries'][modality] for target in dataset['targets']]
            unchanged[modality]=dict(selection_changes=sum(not np.array_equal(
                np.load(system/'uqis/identity_scope/selections'/f'{q}.npy'),
                np.load(baseline/'uqis/identity_scope/selections'/f'{q}.npy')) for q in queries),
                max_rank_difference=max(float(np.max(np.abs(
                    np.load(system/'uqis/identity_scope/predictions'/f'{q}.npy')-
                    np.load(baseline/'uqis/identity_scope/predictions'/f'{q}.npy')))) for q in queries))
        text_queries=[t['queries']['text'] for t in dataset['targets']]
        semantic_difference=max(float(np.max(np.abs(
            np.load(system/'uqis/native_semantic/predictions'/f'{q}.npy')-
            np.load(baseline/'uqis/native_semantic/predictions'/f'{q}.npy')))) for q in text_queries)
        report['conditions'][condition]=dict(metrics=metrics,majority_identity_seeds=sum(r['seed_purity']>.5 for r in cases),
            majority_candidate_in_top={str(k):sum(r['first_majority_candidate_rank'] is not None and
                r['first_majority_candidate_rank']<=k for r in cases) for k in [1,5,10,50,100]},
            dominant_identity={k:sum(r['identity_error_type']==k for r in cases) for k in ['correct','same_class_distractor','other_or_unlabelled']},
            all_four_success=sum(t['all_four_iou_at_least_050'] for t in result['explicit_selection']['paired_query_robustness']['per_target']),
            unchanged_query_controls=unchanged,native_semantic_max_difference=semantic_difference,cases=cases)
    native_cases={r['query_id']:r for r in report['conditions']['native']['cases']}
    native_scores={r['query_id']:r['iou'] for r in results['native']['explicit_selection']['per_query'] if r['modality']=='text'}
    counts={scene:sum(r['scene_id']==scene for r in native_cases.values()) for scene in base_maps}
    for condition in CONDITIONS:
        scores={r['query_id']:r['iou'] for r in results[condition]['explicit_selection']['per_query'] if r['modality']=='text'}
        transitions={key:dict(queries=0,iou_delta_contribution_pp=0.) for key in ['wrong_to_correct','correct_to_wrong','correct_both','wrong_both']}
        for row in report['conditions'][condition]['cases']:
            qid=row['query_id'];before=native_cases[qid]['identity_error_type']=='correct';after=row['identity_error_type']=='correct'
            key=('correct_both' if before else 'wrong_to_correct') if after else ('correct_to_wrong' if before else 'wrong_both')
            transitions[key]['queries']+=1
            transitions[key]['iou_delta_contribution_pp']+=100*(scores[qid]-native_scores[qid])/(len(counts)*counts[row['scene_id']])
        report['conditions'][condition]['dominant_seed_identity_transitions']=transitions
    report['matched_comparisons']={}
    for pair in pairs:
        candidate,control=pair.split(':')
        a=results[candidate]['explicit_selection']['modalities']['text']['per_scene']
        b=results[control]['explicit_selection']['modalities']['text']['per_scene']
        delta={scene:100*(a[scene]['iou']-b[scene]['iou']) for scene in a}
        old={row['query_id']:row['identity_error_type']=='correct' for row in report['conditions'][control]['cases']}
        new={row['query_id']:row['identity_error_type']=='correct' for row in report['conditions'][candidate]['cases']}
        report['matched_comparisons'][pair]=dict(text_iou_delta_pp=scene_clustered_bootstrap(delta),per_scene_delta_pp=delta,
            improved_scenes=sum(v>1e-10 for v in delta.values()),degraded_scenes=sum(v< -1e-10 for v in delta.values()),
            corrected_seed_identities=sum(new[q] and not old[q] for q in old),damaged_seed_identities=sum(old[q] and not new[q] for q in old),
            text_ap_delta_pp=100*(report['conditions'][candidate]['metrics']['text']['ap']-report['conditions'][control]['metrics']['text']['ap']),
            all_four_success_delta=report['conditions'][candidate]['all_four_success']-report['conditions'][control]['all_four_success'])
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({n:{k:v for k,v in d.items() if k!='cases'} for n,d in report['conditions'].items()},indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--conditions',nargs='+',default=CONDITIONS);p.add_argument('--full-condition',default='scene_identity')
    p.add_argument('--pairs',nargs='*',default=[])
    p.add_argument('--declaration',default='configs/recipes/real_referent_grounding.json');a=p.parse_args()
    ROOT=a.root;CONDITIONS=a.conditions;main(a.full_condition,a.declaration,a.pairs)
