"""Complete fixed union response and actual-set attribution, no model selection."""
from pathlib import Path
import numpy as np
from scripts.research.run_query_field_text_union import ROOT,PREVIOUS,read,write,receipt,design
from scripts.research.summarize_query_field_relations import paired
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask


def main():
    d=design();read(ROOT/'scoring_complete.json');per={};records=[]
    for job in read(d['configuration']['path'])['jobs']:
        report=read(ROOT/'predictions'/job['name']/'evaluation/evaluation.json')
        for control,m in report['metrics'].items():
            arm=('parent_native' if control=='native_generic_relevance' else 'parent') if job['name'].endswith('__parent') else control
            result=dict(response_iou=m['iou'],ap=m['ap'],selection_iou=m['selected']['miou'])
            precision=[];recall=[]
            for item in m['selected']['objects']:
                if item['missing']:raise ValueError('missing actual subset')
                target=load_binary_mask(Path(item['gt_path']),threshold=10)
                selected=load_binary_mask(Path(item['pred_path']),threshold=10,grayscale=True)
                tp=int((target&selected).sum());union=int((target|selected).sum())
                if abs(tp/max(union,1)-item['iou'])>1e-12:raise ValueError('selection metric binding differs')
                p=tp/max(int(selected.sum()),1);r=tp/max(int(target.sum()),1)
                precision.append(p);recall.append(r)
                records.append(dict(arm=arm,scene=job['scene'],frame=item['frame'],query=item['query'],iou=item['iou'],precision=p,recall=r))
            result.update(precision=float(np.mean(precision)),recall=float(np.mean(recall)))
            for k,v in result.items():per.setdefault(arm,{}).setdefault(k,{})[job['scene']]=v
    macro={a:{k:float(np.mean(list(v.values()))) for k,v in metrics.items()} for a,metrics in per.items()}
    pairs=[('parent','measure_typed'),('mean_prior','measure_prior'),('mean_typed','measure_typed'),
           ('mean_prior','mean_typed'),('measure_prior','measure_typed')]
    comparisons={a+'__to__'+b:{k:paired(per[a][k],per[b][k],k) for k in per[a]} for a,b in pairs}
    construction=[read(ROOT/'fits'/j['name']/'construction.json') for j in d['jobs']]
    previous=read(PREVIOUS/'summary.json')
    baseline=previous['macro']['confidence_mixture_function']
    report=dict(complete=True,primary=d['primary'],scenes=4,macro=macro,per_scene=per,comparisons=comparisons,
        records=records,construction=construction,frames=sum(c['frames'] for c in construction),
        atoms=sum(c['atoms'] for c in construction),tensor_bytes=sum(c['tensor_bytes'] for c in construction),
        parent_tensor_bytes=sum(c['parent_tensor_bytes'] for c in construction),
        source_replay=dict(cases=sum(len(c['source_replay']) for c in construction),
            maximum_error=max(r['maximum_error'] for c in construction for r in c['source_replay']),
            selection_differences=sum(r['selection_differences'] for c in construction for r in c['source_replay'])),
        parent_recomputed_macro_difference=dict(response_iou=macro['parent']['response_iou']-baseline['LERF2D IoU'],
            selection_iou=macro['parent']['selection_iou']-baseline['LERF actual subset projection IoU']),
        v10_targets={k:v for k,v in previous['v10_targets'].items() if k.startswith('LERF')},
        whole_five_task=False,model_promotion=False,training_steps=0,source_budget_matched_with_parent=False,
        controls_match_added_state=True,source_text_detection_transferred=False,legacy_query_calibration_reused=False,
        implementation_commit=d['implementation_commit'],evidence=[receipt(ROOT/p) for p in
            ['design.json','construction_complete.json','predictions_complete.json','scoring_complete.json']]+[receipt(Path(__file__))])
    write(ROOT/'summary.json',report);write('docs/experiments/2026-10-09-text-union-functions-results.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=['parent','mean_prior','mean_typed','measure_prior','measure_typed']
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    for ax,key,target in zip(axes,['response_iou','selection_iou'],['LERF2D IoU','LERF actual subset projection IoU']):
        values=[100*macro[a][key] for a in arms]
        ax.bar(range(5),values,color=['#888888','#bad1c2','#82a58e','#d6ac60','#287958'])
        ax.axhline(100*report['v10_targets'][target],ls='--',color='#333333')
        for i,v in enumerate(values):ax.text(i,v+.7,f'{v:.2f}',ha='center')
        ax.set_xticks(range(5),['Parent','Mean','Mean+member','Measure','Measure+member'],rotation=25,ha='right')
        ax.set_ylabel('Scene macro IoU (%)');ax.set_ylim(0,90);ax.set_title(target)
    fig.suptitle('Independent union; same added state; dashed: source-assisted v10 reference')
    out=ROOT/'diagnostics';out.mkdir(exist_ok=True);fig.savefig(out/'text_union.png',dpi=180);fig.savefig(out/'text_union.pdf')
    print(macro,flush=True)


if __name__=='__main__':main()
