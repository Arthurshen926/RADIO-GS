"""All-five feedback and paired scene effects; never choose per-task winners."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.research.replay_tmm_response import STUDY


def paired(values, weights=None):
    x=np.asarray(values,dtype=np.float64)*100
    weights=np.ones(len(x)) if weights is None else np.asarray(weights,dtype=np.float64)
    rng=np.random.default_rng(20261016)
    indices=rng.integers(0,len(x),size=(2000,len(x)))
    draws=(x[indices]*weights[indices]).sum(1)/weights[indices].sum(1)
    return dict(delta_pp=float(np.average(x,weights=weights)),scene_bootstrap95_pp=np.quantile(draws,[.025,.975]).tolist(),
        improved_scenes=int((x>1e-8).sum()),degraded_scenes=int((x < -1e-8).sum()),
        scene_count=len(x),unit='scene; horns requests clustered together',samples=2000)


def metric_effect(family,key,arm,reference,summary,uqis):
    if family=='uqis':
        modality=key.removesuffix('_ap')
        original=uqis[reference]['modalities'][modality]['per_scene']
        changed=uqis[arm]['modalities'][modality]['per_scene']
        return paired([changed[s]['average_precision']-v['average_precision'] for s,v in original.items()])
    scenes={}
    for r in summary['rows']:
        if r['job']['benchmark']!=family:continue
        name=r['job']['name'];name='horns' if name.startswith('horns_') else name
        scenes.setdefault(name,[]).append(r['metrics'][arm][key]-r['metrics'][reference][key])
    return paired([np.mean(v) for v in scenes.values()],[len(v) for v in scenes.values()])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='post_response_v1')
    p.add_argument('--compare',action='store_true');a=p.parse_args()
    if a.compare:
        compare();return
    root=STUDY/a.run; summary=json.loads((root/'summary.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    if not summary['complete']:raise ValueError('full cohort is incomplete')
    arms=[arm for arm in summary['macro']['lerf'] if arm!='baseline']
    metrics=[('lerf','semantic_iou','LERF2D IoU',False),('lerf','ap','LERF2D AP',False),
        ('lerf','selected_projection_iou','LERF selected projection IoU',True),
        ('nvos','iou','NVOS IoU',False),('nvos','ap','NVOS AP',False),
        ('scannet','miou_19','ScanNet mIoU19',False),('scannet','miou_15','ScanNet mIoU15',False),
        ('scannet','miou_10','ScanNet mIoU10',False)]
    for m in ['text','image','point_2d','point_3d']:
        metrics.extend([('uqis',m+'_ap','UQIS '+m+' AP',False),
            ('uqis',m+'_selected_iou','UQIS '+m+' actual IoU',True)])
    rows=[]
    uqis={arm:json.loads((root/f'uqis_{arm}.json').read_text()) for arm in ['baseline']+arms}
    for family,key,label,frozen in metrics:
        macro=summary['macro'][family]
        row=dict(family=family,metric=key,label=label,independent_set_head_frozen=frozen,
            values_percent={arm:100*m[key] for arm,m in macro.items()},effects={})
        for arm in arms:
            if frozen:
                row['effects'][arm]=dict(delta_pp=0.,status='unchanged independent selected-set output')
            else:
                row['effects'][arm]=metric_effect(family,key,arm,'baseline',summary,uqis)
        if arms==['scope','matched'] and not frozen:
            row['scope_vs_matched']=metric_effect(family,key,'scope','matched',summary,uqis)
        if arms==['diagonal','rendered_squared','rendered_huber']:
            row['intervention_applied']=family!='uqis' and not frozen
            if not frozen:
                row['rendered_squared_vs_diagonal']=metric_effect(
                    family,key,'rendered_squared','diagonal',summary,uqis)
                row['huber_vs_squared']=metric_effect(
                    family,key,'rendered_huber','rendered_squared',summary,uqis)
        rows.append(row)
    damage={}
    for family in ['lerf','nvos','scannet']:
        damage[family]={arm:dict(corrected=0,harmed=0) for arm in arms}
        for r in summary['rows']:
            if r['job']['benchmark']!=family:continue
            if family=='lerf':
                d=json.loads((STUDY/(a.run+'_lerf_'+r['job']['name'])/'query_results.json').read_text())
                changes={arm:dict(corrected=sum(v['corrected_pixels'] for v in d[arm]),harmed=sum(v['harmed_pixels'] for v in d[arm]))
                    for arm in arms}
            elif family=='nvos':changes=r['pixel_corrections']
            else:changes={arm:r['point_corrections'][arm]['19'] for arm in arms}
            for arm,v in changes.items():
                for key in ['corrected','harmed']:damage[family][arm][key]+=v[key]
    record=dict(schema='tmm.response_refinement.all_five_feedback.v1',run=a.run,rows=rows,
        corrections=damage,correction_units=dict(lerf='query pixels',nvos='pixels',scannet='valid19-class points'),
        summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
        independent_test=False,default_changed=False,query_or_task_winner_selection=False,
        experimental_input=manifest.get('experimental_input','retained frozen v10 responses; no fresh end-to-end claim'),
        controlled_intervention=manifest['changed_variable'],
        confidence_intervals='opened development cohort; descriptive paired scene bootstrap',
        nvos_scene_cluster_note='resample seven scene clusters; two horns requests stay together; weighted estimate retains eight-request mean')
    target=Path('paper/tmm_response');target.mkdir(exist_ok=True)
    (target/f'{a.run}_feedback.json').write_text(json.dumps(record,indent=2)+'\n')
    lines=[r'\begin{tabular}{l'+('r'*(1+len(arms)))+'}',r'\toprule',
        'Task metric & Frozen v10 & '+' & '.join(a.replace('_',' ').capitalize() for a in arms)+r' \\',r'\midrule']
    for r in rows:
        v=r['values_percent']; label=r['label'].replace('_',r'\_')+(' (fixed)' if r['independent_set_head_frozen'] else '')
        lines.append(label+' & '+' & '.join(f'{v[a]:.2f}' for a in ['baseline']+arms)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (target/'tables'/f'{a.run}.tex').write_text('\n'.join(lines)+'\n')
    if arms==['scope','matched']:
        previous=json.loads((target/'post_response_v2_feedback.json').read_text())
        before={(r['family'],r['metric']):r['values_percent'] for r in previous['rows']}
        lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
            r'Metric (\%) & Frozen v10 & Observable & Scope & Matched \\',r'\midrule']
        for row in rows:
            if row['independent_set_head_frozen']:continue
            old=before[(row['family'],row['metric'])];v=row['values_percent']
            if abs(old['baseline']-v['baseline'])>1e-8:raise ValueError('reference changed')
            lines.append(row['label'].replace('_',r'\_')+' & '+' & '.join(f'{value:.2f}' for value in
                [v['baseline'],old['observable'],v['scope'],v['matched']])+r' \\')
        lines.extend([r'\bottomrule',r'\end{tabular}'])
        (target/'tables'/'support_agreement_control.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(rows=rows,corrections=damage),indent=2))


def compare():
    target=Path('paper/tmm_response')
    paths=[target/f'post_response_v{i}_feedback.json' for i in [1,2]]
    first,second=[json.loads(p.read_text()) for p in paths]
    later={(r['family'],r['metric']):r for r in second['rows']}
    rows=[]
    for row in first['rows']:
        other=later.pop((row['family'],row['metric']))
        if abs(row['values_percent']['baseline']-other['values_percent']['baseline'])>1e-8:
            raise ValueError('baseline differs between controlled studies')
        if row['independent_set_head_frozen']!=other['independent_set_head_frozen']:
            raise ValueError('selected-head policy differs')
        rows.append(dict(row,values_percent={**row['values_percent'],**other['values_percent']},
            effects={**row['effects'],**other['effects']}))
    if later:raise ValueError('cohort metrics differ')
    record=dict(schema='tmm.response_refinement.control_comparison.v1',rows=rows,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        independent_test=False,default_changed=False,task_winner_selection=False,
        second_candidate_designed_after_first_feedback=True,
        selection_heads_frozen=True)
    (target/'response_control_comparison.json').write_text(json.dumps(record,indent=2)+'\n')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
        r'Metric (\%) & Frozen v10 & Active partial & Complement & Observable \\',r'\midrule']
    for row in rows:
        if row['independent_set_head_frozen']:continue
        label=row['label'].replace('_',r'\_')
        lines.append(label+' & '+' & '.join(f"{row['values_percent'][arm]:.2f}"
            for arm in ['baseline','partial','complement','observable'])+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (target/'tables'/'response_control_comparison.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
