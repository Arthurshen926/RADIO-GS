"""All-arm results and complete point attribution after the scoring barrier."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from scripts.research.run_query_field_joint_observation import ROOT,PREVIOUS,ARMS,read,write,receipt,frozen_design,verify_predictions
from scripts.research.summarize_query_field_relations import paired


def diagnose():
    read(ROOT/'scoring_complete.json');design=frozen_design()
    cohort=read(design['arms'][ARMS[0]]['path']);package=Path(cohort['uqis_package'])
    targets=read(package/'evaluator/targets.json');records=[];evidence=[]
    previous=read(PREVIOUS/'summary.json')
    known={q['query_id'] for s in previous['point_activation']['scenes'] for q in s['queries'] if q['correct_registered_support']}
    native={a:dict(queries=0,exact=0,max_error=0.) for a in ARMS}
    for job in design['fields']:
        scene=job['scene'];domain=next(d for d in targets['scene_domains'] if d['scene_id']==scene)
        source=read(Path(job['observations'])/'observations.json')
        geometry=torch.load(source['geometry']['path'],map_location='cpu',weights_only=True)
        xyz=geometry['state_dict']['xyz'].numpy();source_ids=geometry['source_vertex_ids'].numpy()
        mesh=np.load(package/'method'/domain['mesh_xyz_path'])
        labels=np.load(package/'evaluator'/domain['mesh_instance_ids_path'])
        if not np.array_equal(xyz,mesh[source_ids]):raise ValueError('diagnostic carrier mapping changed')
        distance,index=cKDTree(xyz).query(mesh,workers=4);row_labels=labels[source_ids]
        paths={a:ROOT/a/'predictions/uqis' for a in ARMS}
        paths['confidence_mixture_function']=PREVIOUS/'confidence_mixture_function/predictions/uqis'
        receipts={a:read(p/'receipts'/(scene+'.json')) for a,p in paths.items()}
        for q in receipts[ARMS[0]]['queries']:
            if q['modality'] not in ['text','image']:continue
            original=torch.load(paths['confidence_mixture_function']/'gaussians'/(q['query_id']+'.pt'),map_location='cpu',weights_only=True)['native_scores']
            for a in ARMS:
                actual=torch.load(paths[a]/'gaussians'/(q['query_id']+'.pt'),map_location='cpu',weights_only=True)['native_scores']
                native[a]['queries']+=1;native[a]['exact']+=int(torch.equal(original,actual))
                native[a]['max_error']=max(native[a]['max_error'],float((actual-original).abs().max()))
        for target in [t for t in targets['targets'] if t['scene_id']==scene]:
            qid=target['queries']['point_3d'];truth=labels==target['instance_id'];case=dict(scene=scene,query_id=qid,
                prior_known_correct=qid in known,target_vertices=int(truth.sum()),arms={})
            previous_selected=np.load(paths['confidence_mixture_function']/'selections'/(qid+'.npy'))
            for a in ARMS:
                data=torch.load(paths[a]/'gaussians'/(qid+'.pt'),map_location='cpu',weights_only=True)
                selected=data['selected'].numpy()[index];ids=data['candidates'].numpy()
                tp=int((selected&truth).sum());fp=int((selected&~truth).sum());fn=int((~selected&truth).sum())
                case['arms'][a]=dict(correct_registered_support=bool(len(ids) and (row_labels[ids]==target['instance_id']).all()),
                    unsupported=not len(ids), precision=tp/max(tp+fp,1), recall=tp/max(tp+fn,1),iou=tp/max(tp+fp+fn,1),
                    selected_vertices=int(selected.sum()),gained_target=int((selected&~previous_selected&truth).sum()),
                    lost_target=int((~selected&previous_selected&truth).sum()),gained_other=int((selected&~previous_selected&~truth).sum()),
                    removed_other=int((~selected&previous_selected&~truth).sum()))
            records.append(case)
        evidence.extend([receipt(package/'method'/domain['mesh_xyz_path']),receipt(package/'evaluator'/domain['mesh_instance_ids_path'])])
        print('diagnosed',scene,flush=True)
    if len(records)!=67 or sum(q['prior_known_correct'] for q in records)!=59:raise ValueError('diagnostic cohort differs')
    summary={}
    for a in ARMS:
        values=[q['arms'][a] for q in records if q['prior_known_correct']]
        summary[a]=dict(fixed_correct_supports=59,**{k:float(np.mean([q[k] for q in values])) for k in ['precision','recall','iou']},
            **{k:sum(q[k] for q in values) for k in ['gained_target','lost_target','gained_other','removed_other']},
            total_correct_supports=sum(q['arms'][a]['correct_registered_support'] for q in records),
            unsupported=sum(q['arms'][a]['unsupported'] for q in records))
    if any(v['queries']!=134 for v in native.values()):raise ValueError('native input audit incomplete')
    write(ROOT/'diagnostics/actual_queries.json',dict(records=records,summary=summary,native=native,
        target_labels_used='post-score diagnostics only',model_change=False,
        aggregation='fixed59 correct-query means and separate pooled transitions; not benchmark scene macro',
        code=receipt(Path(__file__)),evidence=evidence))


def summarize():
    design=frozen_design();read(ROOT/'scoring_complete.json');verify_predictions(design)
    previous=read(PREVIOUS/'summary.json');data={};paths=[]
    for a in ARMS:
        path=ROOT/a/'predictions/uqis/evaluation.json';d=read(path);paths.append(path)
        if d['query_count']!=268 or d['target_count']!=67:raise ValueError('score denominator differs')
        data[a]={}
        for m in ['text','image','point_2d','point_3d']:
            data[a]['UQIS '+m+' IoU']={s:v['iou'] for s,v in d['explicit_selection']['modalities'][m]['per_scene'].items()}
            data[a]['UQIS '+m+' AP']={s:v['average_precision'] for s,v in d['modalities'][m]['per_scene'].items()}
    for a in ['protected_training','soft_paired_function','confidence_mixture_function']:
        data[a]={k:v for k,v in previous['per_scene'][a].items() if k.startswith('UQIS')}
    comparisons={}
    for a,b in [('small_source','full_source'),('small_supervised','full_supervised'),
                ('small_source','small_supervised'),('full_source','full_supervised'),
                *[('confidence_mixture_function',a) for a in ARMS]]:
        comparisons[a+'__to__'+b]={k:paired(data[a][k],data[b][k],k) for k in data[b]}
    interactions={}
    for k in data[ARMS[0]]:
        low={s:data['small_supervised'][k][s]-data['small_source'][k][s] for s in data[ARMS[0]][k]}
        high={s:data['full_supervised'][k][s]-data['full_source'][k][s] for s in low}
        interactions[k]=paired(low,high,k)
    construction=[read(ROOT/'fits'/j['scene']/'construction.json') for j in design['fields']]
    if len(construction)!=9 or any({v['arm'] for v in r['records']}!=set(ARMS) for r in construction):
        raise ValueError('construction incomplete')
    replay=[q for r in construction for arm in r['records'] if arm['arm'].endswith('_source') for q in arm['source_replay']]
    diagnostic=read(ROOT/'diagnostics/actual_queries.json');cold=read(ROOT/'validation/runtime/report.json')
    remaining=read(ROOT/'diagnostics/remaining_family_sources.json')
    tests=read(ROOT/'validation/focused_tests.json')
    native_transfer=read(ROOT/'diagnostics/legacy_native_transfer.json')
    supervision_split=read(ROOT/'diagnostics/supervision_disjointness.json')
    evidence=paths+[ROOT/p for p in ['design.json','construction_complete.json','predictions_complete.json',
        'scoring_complete.json','diagnostics/actual_queries.json','validation/runtime/report.json',
        'diagnostics/remaining_family_sources.json','validation/focused_tests.json',
        'diagnostics/legacy_native_transfer.json','diagnostics/supervision_disjointness.json']]
    evidence += [ROOT/'fits'/j['scene']/'construction.json' for j in design['fields']]+[Path(__file__)]
    result=dict(complete=True,completion_scope=design['scope'],model_promotion=False,
        construction_commit=design['implementation_commit'],per_scene=data,
        macro={a:{k:float(np.mean(list(v.values()))) for k,v in metrics.items()} for a,metrics in data.items()},
        comparisons=comparisons,factorial_interaction=interactions,
        budgets={b:sum(len(j['budgets'][b]) for j in design['fields']) for b in ['small','full']},
        source_replay=dict(cases=len(replay),rows=sum(q['rows'] for q in replay),
            max_feature_error=max(q['feature_error'] for q in replay),max_source_function_error=max(q['source_only_error'] for q in replay),
            max_forest_operator_error=max(q['forest_operator_error'] for q in replay),
            max_compiled_forest_function_error=max(q['forest_function_error'] for q in replay),
            selection_disagreements=sum(q['selection_disagreements'] for q in replay)),
        tensor_bytes={a:sum(next(v['tensor_bytes'] for v in r['records'] if v['arm']==a) for r in construction) for a in ARMS},
        actual_queries=diagnostic,runtime=cold,construction=construction,
        focused_tests=tests,legacy_native_transfer=native_transfer,
        supervision_disjointness=supervision_split,
        remaining_family_sources=remaining['summary'],
        remaining_source_inventory_scope=remaining['limitation'],
        strict_cold_replay_passed=cold['strict_bitwise_passed'],
        numerical_cold_replay_passed=cold['numerical_and_selection_passed'],
        source_denial_passed=cold['source_denial_passed'],
        supervision_conditions={a:'independent22-room3D instance labels,3-room calibration' if a.endswith('supervised') else 'source-only' for a in ARMS},
        v10_targets={k:v for k,v in previous['v10_targets'].items() if k.startswith('UQIS')},
        whole_five_task_candidate=False,independent_final_test=False,native_patch_function_transferred=False,
        evidence=[receipt(p) for p in evidence])
    write(ROOT/'summary.json',result);write('docs/experiments/2026-10-09-joint-observation-transfer-results.json',result)
    print(json_compact(result['macro']),flush=True)


def json_compact(value):
    import json
    return json.dumps(value,indent=2)


def plot():
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=read(ROOT/'summary.json');out=ROOT/'diagnostics';table=[]
    arms=['confidence_mixture_function','small_source','full_source','small_supervised','full_supervised']
    labels=['Prior','18 / src','Full / src','18 / 3D','Full / 3D']
    fig,axes=plt.subplots(1,4,figsize=(14,4.4),constrained_layout=True)
    for ax,modality in zip(axes,['text','image','point_2d','point_3d']):
        key='UQIS '+modality+' IoU';values=[100*report['macro'][a][key] for a in arms]
        target=100*report['v10_targets'][key]
        ax.bar(range(5),values,color=['#888888','#8ab8a2','#327654','#dab469','#a8601e'])
        ax.axhline(target,color='#333333',linestyle='--')
        for i,value in enumerate(values):ax.text(i,value+.8,f'{value:.2f}',ha='center',fontsize=8)
        ax.set_xticks(range(5),labels,rotation=25,ha='right');ax.set_title('UQIS '+modality.replace('_',' '))
        ax.set_ylim(0,min(100,max(target,*values)+12));ax.set_ylabel('Scene macro IoU (%)')
        ax.spines[['top','right']].set_visible(False)
        table.append(dict(metric=key,**dict(zip(arms,values)),v10=target))
    fig.suptitle('Observation budget and member decision: fixed complete UQIS development cohort')
    fig.supxlabel('src: source-only conditional vote. 3D: existing22-room instance supervision /3-room calibration.\n'
        'Full excludes the same two development views per scene. Dashed: v10; source/input/native identity differ.',fontsize=9)
    for suffix in ['png','pdf']:fig.savefig(out/('factorial_metrics.'+suffix),dpi=180)
    plt.close(fig)
    with (out/'factorial_metrics.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    write(out/'factorial_metrics.json',dict(summary=receipt(ROOT/'summary.json'),table=table,
        artifacts=[receipt(out/('factorial_metrics.'+suffix)) for suffix in ['png','pdf','csv']]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['diagnose','summarize','plot']);a=p.parse_args()
    torch.set_num_threads(4)
    {'diagnose':diagnose,'summarize':summarize,'plot':plot}[a.stage]()
