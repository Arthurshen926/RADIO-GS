"""Measure completeness and false additions on the unchanged full query cohort."""
import argparse
import json
from pathlib import Path

import numpy as np

from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('output/witnessed_scope_study'))
    p.add_argument('--entity-system',type=Path)
    p.add_argument('--panoptic-system',type=Path)
    args=p.parse_args()
    roots={'v8':Path('output/system_v8'), **{m:args.root/m/'system' for m in ['majority','union']}}
    if args.entity_system:roots['entity']=args.entity_system
    if args.panoptic_system:roots['panoptic']=args.panoptic_system
    results={name:read(root/'uqis/identity_scope/result.json') for name,root in roots.items()}
    baseline=results['v8'];base_rows={r['query_id']:r for r in baseline['explicit_selection']['per_query']}
    rows={}
    for name,result in results.items():
        row=dict(modalities={},paired=result['explicit_selection']['paired_query_robustness'])
        for modality,values in result['modalities'].items():
            selected=result['explicit_selection']['modalities'][modality]
            differences={s:100*(v['iou']-baseline['explicit_selection']['modalities'][modality]['per_scene'][s]['iou'])
                         for s,v in selected['per_scene'].items()}
            queries=[r for r in result['explicit_selection']['per_query'] if r['modality']==modality]
            stable=[r for r in queries if r['target_is_unique_dominant_instance'] and
                    base_rows[r['query_id']]['target_is_unique_dominant_instance']]
            row['modalities'][modality]=dict(ap=100*values['scene_macro']['average_precision'],
                **{k:100*v for k,v in selected['scene_macro'].items()},
                iou_gain_pp=scene_clustered_bootstrap(differences),per_scene_iou_gain_pp=differences,
                target_dominant=sum(r['target_is_unique_dominant_instance'] for r in queries),
                all_success_050=sum(r['iou']>=.5 for r in queries),
                identity_corrected=sum(r['target_is_unique_dominant_instance'] and not base_rows[r['query_id']]['target_is_unique_dominant_instance'] for r in queries),
                identity_broken=sum(not r['target_is_unique_dominant_instance'] and base_rows[r['query_id']]['target_is_unique_dominant_instance'] for r in queries),
                matched_identity_condition=dict(queries=len(stable),aggregation='query macro on identical subset, not full scene macro',
                    **{k:dict(before=100*float(np.mean([base_rows[r['query_id']][k] for r in stable])),
                              after=100*float(np.mean([r[k] for r in stable]))) for k in ['iou','coverage','purity']}),
                empty_selections=selected['empty_selections'])
        if name!='v8':
            row['control_parity']={}
            for control in ['native_semantic','anchor_comembership','identity_region']:
                files=list((roots[name]/'uqis'/control/'predictions').glob('*.npy'))
                row['control_parity'][control]=dict(queries=len(files),max_absolute_difference=max(
                    float(np.abs(np.load(f)-np.load(roots['v8']/'uqis'/control/'predictions'/f.name)).max()) for f in files))
            reports=[read(f) for f in (roots[name]/'uqis').glob('*/query/registered/runs/*.json')]
            row['cost']=dict(query_seconds_sum=sum(r['seconds'] for r in reports),map_mib=sum(r['map_bytes'] for r in reports)/2**20)
            row['seed_changes']=sum(q['instance_identity']['seed_region']!=
                (q['witnessed_scope']['seed_region'] if 'witnessed_scope' in q else q['entity_scope']['native_seed_region'])
                for report in reports for q in report['query_details'] if q['modality'] in ['text','image'])
            if name in ['entity','panoptic']:
                row['entity_applied']={m:sum(q['entity_scope']['applied'] for report in reports for q in report['query_details']
                    if q['modality']==m) for m in result['modalities']}
        rows[name]=row
    builds=[read(f) for f in (args.root/'maps').glob('*/build.json')]
    summary=dict(results=rows,compile_cost=dict(seconds=sum(b['seconds'] for b in builds),
        directed_edges=sum(b['directed_edges'] for b in builds),witness_regions=sum(b['witness_regions'] for b in builds)),
        completed_scene_jobs=9*(len(roots)-1),completed_query_executions=268*(len(roots)-1),
        all_query_selections={name:r['explicit_selection']['per_query'] for name,r in results.items()},
        independent_test=False,public_benchmarks_reexecuted=False)
    (args.root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(rows);modalities=list(baseline['modalities']);x=np.arange(len(modalities));width=.8/len(names)
    figure,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,metric in zip(axes,['iou','coverage','purity']):
        for i,name in enumerate(names):
            ax.bar(x+(i-(len(names)-1)/2)*width,
                [rows[name]['modalities'][m][metric] for m in modalities],width,label=name)
        ax.set_xticks(x,modalities);ax.set_title('Selected set '+metric+' (%)');ax.set_ylim(0,100)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    figure.suptitle('Same 9 development scenes / 67 targets / 268 independent queries per condition')
    figure.savefig(args.root/'comparison.png',dpi=160);plt.close(figure)
    print(json.dumps({name:{m:{k:v[k] for k in ['ap','iou','purity','coverage','target_dominant']} for m,v in row['modalities'].items()}
        for name,row in rows.items()},indent=2))


if __name__=='__main__':main()
