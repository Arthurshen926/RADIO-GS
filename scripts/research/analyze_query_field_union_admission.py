"""Report exploratory response changes and exact admission changes, without refit."""
from pathlib import Path
import shutil
import torch
from scripts.research.run_query_field_union_admission import ROOT,SOURCE,read,write,receipt
from scripts.research.summarize_query_field_relations import paired


def main():
    old=read(SOURCE/'summary.json');new=read(ROOT/'summary.json');cold=read(ROOT/'validation/runtime/report.json')
    rows=[];prediction_receipts=[]
    for scene in new['per_scene']:
        path=SOURCE/'predictions'/(scene+'__measure')/'predictions.pt';data=torch.load(path,weights_only=True)
        original=data['selected_by_control']['measure_typed'];prior=data['selected_by_control']['measure_prior']
        if bool((prior&~original).any()):raise ValueError('old positive support unexpectedly removed admission')
        for i,text in enumerate(data['texts']):
            rows.append(dict(scene=scene,query=text,old_selected=int(original[:,i].sum()),
                semantic_selected=int(prior[:,i].sum()),unsupported_rank_admissions=int((original[:,i]&~prior[:,i]).sum())))
        prediction_receipts.append(receipt(path))
    comparisons={}
    for arm in ['parent','measure_prior','measure_typed']:
        comparisons[arm]={k:paired(old['per_scene'][arm][k],{s:r[k] for s,r in new['per_scene'].items()},k)
            for k in ['response_iou','selection_iou','ap']}
    report=dict(complete=True,exploratory=True,previous_labels_seen=True,comparisons=comparisons,queries=len(rows),records=rows,
        removed_rank_only_admissions=sum(r['unsupported_rank_admissions'] for r in rows),
        affected_queries=sum(r['unsupported_rank_admissions']>0 for r in rows),
        counted_admissions_are_gaussian_query_pairs_not_GT_false_positives=True,
        source=receipt(SOURCE/'summary.json'),new=receipt(ROOT/'summary.json'),cold=receipt(ROOT/'validation/runtime/report.json'),
        predictions=prediction_receipts,analysis_code=receipt(Path(__file__)))
    write(ROOT/'attribution.json',report);write('docs/experiments/2026-10-09-union-admission-attribution.json',report)
    shutil.copyfile(ROOT/'validation/runtime/report.json','docs/experiments/2026-10-09-union-admission-runtime.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,key,target,title in zip(axes,['response_iou','selection_iou'],['LERF2D IoU','LERF actual subset projection IoU'],['LERF2D response IoU','Actual-set projection IoU (not 3D GT)']):
        values=[old['macro'][a][key]*100 for a in ['parent','measure_typed','measure_prior']]+[100*new['macro'][key]]
        ax.bar(range(4),values,color=['#888888','#c97a63','#d6ac60','#287958']);ax.axhline(100*old['v10_targets'][target],ls='--',color='#333333')
        for i,v in enumerate(values):ax.text(i,v+.7,f'{v:.2f}',ha='center')
        ax.set_xticks(range(4),['Parent','Failed primary','Semantic prior','Admission repair'],rotation=20,ha='right');ax.set_ylim(0,85);ax.set_ylabel('Scene macro IoU (%)');ax.set_title(title)
    fig.suptitle('Complete four-scene cohort; repair is exploratory; dashed: source-assisted v10')
    out=ROOT/'diagnostics';out.mkdir(exist_ok=False);fig.savefig(out/'admission.png',dpi=180);fig.savefig(out/'admission.pdf')
    print({k:v for k,v in report.items() if k in ['queries','removed_rank_only_admissions','affected_queries','comparisons']},flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);main()
