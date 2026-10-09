"""Report every predeclared categorical function and common-member control."""
import json
from pathlib import Path
import numpy as np
from scripts.research.run_query_field_semantic_measure import ROOT,PREVIOUS,read,write,receipt,design
from scripts.research.summarize_query_field_relations import paired


def main():
    d=design();barrier=read(ROOT/'scoring_complete.json');results=barrier['results']
    if len(results)!=10:raise ValueError('incomplete cohort')
    per={};macro={}
    for arm in results[0]['metrics']:
        per[arm]={}
        for split in results[0]['metrics'][arm]:
            for metric in ['miou','macc']:
                per[arm][split+' '+metric]={r['scene']:r['metrics'][arm][split][metric] for r in results}
        macro[arm]={k:float(np.mean(list(v.values()))) for k,v in per[arm].items()}
    pairs=[('parent','reliable_typed'),('mean_typed','measure_typed'),('measure_typed','reliable_typed'),
           *[(r+'_prior',r+'_typed') for r in ['mean','measure','reliable']]]
    comparisons={a+'__to__'+b:{k:paired(per[a][k],per[b][k],k) for k in per[a]} for a,b in pairs}
    construction=[read(ROOT/'fits'/j['name']/'construction.json') for j in d['jobs']]
    prior=read(PREVIOUS/'summary.json');parent_delta={}
    for split in results[0]['metrics']['parent']:
        key='ScanNet typed mIoU'+''.join(filter(str.isdigit,split))
        parent_delta[key]=macro['parent'][split+' miou']-prior['macro']['confidence_mixture_function'][key]
    report=dict(complete=True,scenes=10,primary=d['primary'],macro=macro,per_scene=per,comparisons=comparisons,
        parent_recomputed_macro_difference=parent_delta,construction=construction,
        source_views=sum(c['frames'] for c in construction),atoms=sum(c['atoms'] for c in construction),
        tensor_bytes=sum(c['tensor_bytes'] for c in construction),parent_tensor_bytes=sum(c['parent_tensor_bytes'] for c in construction),
        source_function_replay=dict(cases=sum(len(c['replay']) for c in construction),
            maximum_error=max(r['maximum_error'] for c in construction for r in c['replay']),
            argmax_differences=sum(r['argmax_differences'] for c in construction for r in c['replay'])),
        v10_targets={k:v for k,v in prior['v10_targets'].items() if k.startswith('ScanNet')},
        no_new_training=True,new_instance_supervision=False,full_five_task=False,model_promotion=False,
        runtime_source_text_detection_transferred=False,local_auxiliary_channel_transferred=False,
        source_budget_matched_with_parent=False,controls_match_all_new_state=True,
        implementation_commit=d['implementation_commit'],evidence=[receipt(ROOT/p) for p in
            ['design.json','construction_complete.json','predictions_complete.json','scoring_complete.json']]+[receipt(Path(__file__))])
    write(ROOT/'summary.json',report)
    write('docs/experiments/2026-10-09-semantic-measure-results.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    arms=['parent','mean_typed','measure_typed','reliable_typed']
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
    for ax,split in zip(axes,results[0]['metrics']['parent']):
        values=[100*macro[a][split+' miou'] for a in arms]
        ax.bar(range(4),values,color=['#888888','#a6bdaa','#d6ac60','#287958'])
        key='ScanNet typed mIoU'+''.join(filter(str.isdigit,split))
        ax.axhline(100*prior['v10_targets'][key],color='#333333',ls='--')
        for i,v in enumerate(values):ax.text(i,v+.5,f'{v:.2f}',ha='center')
        ax.set_xticks(range(4),['Parent','Mean','Measure','Reliable'],rotation=20)
        ax.set_title(split);ax.set_ylabel('Scene macro mIoU (%)');ax.set_ylim(0,85)
    fig.suptitle('Same common member; full semantic measure controls. Dashed: source-assisted v10.')
    path=ROOT/'diagnostics';path.mkdir(exist_ok=True)
    fig.savefig(path/'semantic_measure.png',dpi=180);fig.savefig(path/'semantic_measure.pdf')
    print(json.dumps(macro,indent=2));print('replay',report['source_function_replay'])


if __name__=='__main__':main()
