"""Full-cohort native-function results and post-score identity attribution."""
from pathlib import Path
import argparse

import numpy as np
import torch

from scripts.research.run_query_field_native_modes import ROOT,PARENT,ARMS,read,write,receipt,design
from scripts.research.summarize_query_field_relations import paired


def summarize():
    read(ROOT/'scoring_complete.json');d=design();package=Path(d['package'])
    targets=read(package/'evaluator/targets.json');data={};records=[];invariance={}
    for arm in ARMS:
        result=read(ROOT/arm/'predictions/uqis/evaluation.json')
        if result['query_count']!=268 or result['target_count']!=67:raise ValueError('score denominator differs')
        data[arm]={}
        for m in ['text','image','point_2d','point_3d']:
            data[arm][m+' IoU']={s:v['iou'] for s,v in result['explicit_selection']['modalities'][m]['per_scene'].items()}
            data[arm][m+' AP']={s:v['average_precision'] for s,v in result['modalities'][m]['per_scene'].items()}
        invariance[arm]=dict(native_queries=0,native_exact=0,point_queries=0,point_selection_exact=0,point_max_error=0.)
    prior=read(PARENT/'summary.json')
    data['parent']={k.removeprefix('UQIS '):v for k,v in prior['per_scene']['full_supervised'].items()}
    for job in d['fields']:
        scene=job['scene'];domain=next(x for x in targets['scene_domains'] if x['scene_id']==scene)
        decisions={arm:{q['query_id']:q['diagnostics'] for q in
            read(ROOT/arm/'predictions/uqis/receipts'/(scene+'.json'))['queries']} for arm in ARMS}
        labels=np.load(package/'evaluator'/domain['mesh_instance_ids_path'])
        obs=read(Path(job['observations'])/'observations.json')
        geo=torch.load(obs['geometry']['path'],map_location='cpu',weights_only=True)
        row_labels=labels[geo['source_vertex_ids'].numpy()]
        for target in [t for t in targets['targets'] if t['scene_id']==scene]:
            truth=labels==target['instance_id']
            for m,qid in target['queries'].items():
                for arm in ARMS:
                    path=ROOT/arm/'predictions/uqis';saved=torch.load(path/'gaussians'/(qid+'.pt'),map_location='cpu',weights_only=True)
                    ids=saved['candidates'].numpy();weight=saved['candidate_weights'].numpy();correct=row_labels[ids]==target['instance_id']
                    chosen=decisions[arm][qid].get('chosen_mode',int(weight.argmax()) if len(weight) else -1)
                    selected=np.load(path/'selections'/(qid+'.npy'));tp=int((selected&truth).sum());fp=int((selected&~truth).sum());fn=int((~selected&truth).sum())
                    records.append(dict(arm=arm,scene=scene,modality=m,query_id=qid,candidates=len(ids),
                        correct_identity=bool(len(ids) and correct[chosen]),chosen_candidate=chosen,candidate_has_target=bool(correct.any()),
                        iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1)))
                    if m in ['text','image']:
                        old=torch.load(PARENT/'full_supervised/predictions/uqis/gaussians'/(qid+'.pt'),map_location='cpu',weights_only=True)
                        invariance[arm]['native_queries']+=1
                        invariance[arm]['native_exact']+=int(torch.equal(old['native_scores'],saved['native_scores']))
                    else:
                        oldpath=(Path('output/query_field_20261009/surface_input_function_v1/gaussians') if m=='point_2d' else PARENT/'full_supervised/predictions/uqis/gaussians')
                        old=torch.load(oldpath/(qid+'.pt'),map_location='cpu',weights_only=True)
                        invariance[arm]['point_queries']+=1
                        invariance[arm]['point_selection_exact']+=int(torch.equal(old['selected'],saved['selected']))
                        invariance[arm]['point_max_error']=max(invariance[arm]['point_max_error'],float((old['membership']-saved['membership']).abs().max()))
        print('diagnosed',scene,flush=True)
    attribution={}
    for arm in ARMS:
        attribution[arm]={}
        for m in ['text','image','point_2d','point_3d']:
            rows=[r for r in records if r['arm']==arm and r['modality']==m]
            if len(rows)!=67:raise ValueError('attribution denominator differs')
            attribution[arm][m]=dict(queries=len(rows),correct_identity=sum(r['correct_identity'] for r in rows),
                candidate_has_target=sum(r['candidate_has_target'] for r in rows),
                correct_identity_iou=float(np.mean([r['iou'] for r in rows if r['correct_identity']])) if any(r['correct_identity'] for r in rows) else None)
    construction=[read(ROOT/'fits'/j['scene']/'construction.json') for j in d['fields']]
    output=dict(complete=True,scope='all9 UQIS scenes/all804 inputs;not full five-task validation',
        model_promotion=False,whole_five_task_candidate=False,additional_instance_supervision=True,
        macro={a:{k:float(np.mean(list(v.values()))) for k,v in values.items()} for a,values in data.items()},
        per_scene=data,comparisons={a+'__to__'+b:{k:paired(data[a][k],data[b][k],k) for k in data[a]}
            for a,b in [('mean_native','measure_native'),('measure_resized','measure_native'),*[('parent',a) for a in ARMS]]},
        attribution=attribution,records=records,invariance=invariance,construction=construction,
        source_replay=dict(cases=sum(len(c['source_replay']) for c in construction),
            maximum_error=max(r['maximum_error'] for c in construction for r in c['source_replay']),
            source_anchor_differences=sum(c['source_anchor_differences'] for c in construction)),
        v10_targets=prior['v10_targets'],construction_commit=d['implementation_commit'],
        label_use='post-score diagnostics only; no model change',
        diagnostics_aggregation='query-conditioned descriptions,not causal comparison or scene macro',
        evidence=[receipt(ROOT/p) for p in ['design.json','construction_complete.json','predictions_complete.json','scoring_complete.json']]+[receipt(Path(__file__))])
    write(ROOT/'summary.json',output);write('docs/experiments/2026-10-09-native-mode-functions-results.json',output)
    print(output['macro'],flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=read(ROOT/'summary.json');fig,axes=plt.subplots(1,4,figsize=(14,4.2),constrained_layout=True)
    arms=['parent',*ARMS];labels=['Parent','Mean/native','Measure/448','Measure/native']
    for ax,m in zip(axes,['text','image','point_2d','point_3d']):
        values=[100*d['macro'][a][m+' IoU'] for a in arms]
        target=100*d['v10_targets']['UQIS '+m+' IoU']
        ax.bar(range(4),values,color=['#888888','#a5c1ad','#d6ab61','#317958'])
        ax.axhline(target,color='#333333',linestyle='--')
        for i,v in enumerate(values):ax.text(i,v+.6,f'{v:.2f}',ha='center',fontsize=8)
        ax.set_xticks(range(4),labels,rotation=25,ha='right');ax.set_title('UQIS '+m);ax.set_ylim(0,max(target,*values)+10)
        ax.set_ylabel('Scene macro IoU (%)');ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('All new arms: full legal sources, inherited 22-room supervision / 3-room calibration, surface 2D input.\nDashed: v10 reference. Same Gaussian member function; uncompressed appearance measures.',fontsize=9)
    out=ROOT/'diagnostics';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/('native_modes.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['summarize','plot']);args=p.parse_args();torch.set_num_threads(4)
    {'summarize':summarize,'plot':plot}[args.stage]()
