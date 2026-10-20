"""Build all-eight N03 paired effects and paper tables from verified receipts."""
import csv
import json
from pathlib import Path
from statistics import mean

from radio_gs.response_study.local_credit_study import check_local_credit_study, check_local_credit_spatial, check_reference_scope
from radio_gs.response_study.reference_study import read

REPO=Path(__file__).resolve().parents[2];PAPER=REPO/'paper/tmm_response'
ARMS=['v10','coverage','coverage_anchor_domain','fit_local','held_local','held_uniform']
LABELS=['v10','Coverage','Anchor domain','Fit-local','Held-local','Matched uniform']


def main():
    registry=read(REPO/'configs/research/tmm_response.json')
    item=next(r for r in registry['reference_studies'] if r['id']=='N03')
    record=check_local_credit_study(REPO,REPO/registry['study_root'],item,read(REPO/registry['recipe'])['jobs'])
    if not record['complete']:raise ValueError('N03 has not completed all eight tasks')
    import numpy as np
    rows=record['rows'];effects={}
    comparisons=[('held_local','v10'),('held_local','coverage'),('held_local','held_uniform'),
                 ('held_local','fit_local'),('coverage_anchor_domain','coverage')]
    for track in record['macro']:
        effects[track]={}
        for arm,base in comparisons:
            key=arm+'_vs_'+base;effects[track][key]={}
            for metric in ('iou','average_precision'):
                delta={n:100*(r[arm][track][metric]-r[base][track][metric]) for n,r in rows.items()}
                clusters={n:('horns' if n.startswith('horns_') else n) for n in rows}
                unique=sorted(set(clusters.values()));rng=np.random.default_rng(0)
                boot=[mean(d for c in sampled for n,d in delta.items() if clusters[n]==c)
                      for sampled in rng.choice(unique,size=(5000,len(unique)),replace=True)]
                effects[track][key][metric]=dict(delta_pp=mean(delta.values()),per_task_delta_pp=delta,
                    improved=sum(v>1e-6 for v in delta.values()),harmed=sum(v< -1e-6 for v in delta.values()),
                    unchanged=sum(abs(v)<=1e-6 for v in delta.values()),independent_scene_clusters=7,
                    descriptive_scene_bootstrap95_pp=np.quantile(boot,[.025,.975]).tolist())
    record['effects']=effects
    if (REPO/registry['study_root']/'nvos_local_credit_spatial_v1/summary.json').exists():
        spatial=check_local_credit_spatial(REPO/registry['study_root'])
        record['spatial_diagnosis']=spatial['rows']
        record['receipts'].update(spatial['receipts'])
    scope=check_reference_scope(REPO/registry['study_root'])
    record['reference_scope']=scope['rows'];record['receipts'].update(scope['receipts'])
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Task & Seed--track IoU & Seed retained & Track inside seed \\',r'\midrule']
    for n,r in scope['rows'].items():
        lines.append(n.replace('_',r'\_')+' & '+' & '.join(f"{100*r['agreement'][m]:.2f}" for m in ('iou','recall','precision'))+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/nvos_reference_scope.tex').write_text('\n'.join(lines)+'\n')
    (PAPER/'nvos_local_credit_feedback.json').write_text(json.dumps(record,indent=2)+'\n')
    flat=[dict(task=n,arm=a,gt_track=t,**m) for n,arms in rows.items() for a,tracks in arms.items() for t,m in tracks.items()]
    with (PAPER/'artifacts/nvos_local_credit.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',
           r'Task & v10 & Coverage & Domain & Fit-local & Held-local & Uniform \\',r'\midrule']
    for name,values in rows.items():
        lines.append(name.replace('_',r'\_')+' & '+' & '.join(f"{100*values[a]['retained_gt_gt127']['iou']:.2f}" for a in ARMS)+r' \\')
    for track,label in [('retained_gt_gt127','Mean (retained GT)'),('ludvig_gt_maximum','Mean (LUDVIG GT)')]:
        lines.append(r'\midrule '+label+' & '+' & '.join(f"{100*record['macro'][track][a]['iou']:.2f}" for a in ARMS)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/nvos_local_credit.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Arm & IoU & Precision & Recall & AP \\',r'\midrule']
    for a,label in zip(ARMS,LABELS):
        lines.append(label+' & '+' & '.join(f"{100*record['macro']['retained_gt_gt127'][a][m]:.2f}" for m in ('iou','precision','recall','average_precision'))+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/nvos_local_credit_macro.tex').write_text('\n'.join(lines)+'\n')
    values=record['macro']['retained_gt_gt127'];primary=values['held_local']
    dc=100*(primary['iou']-values['coverage']['iou']);du=100*(primary['iou']-values['held_uniform']['iou'])
    text=(r'\paragraph{Local response edits with a matched control.}'+'\n'
        +f"N03 fixes six arms on all eight tasks and designates held-source local credit before scoring. "
        +f"Its IoU is {100*primary['iou']:.4f}\\% and AP {100*primary['average_precision']:.4f}\\%, "
        +f"with IoU differences {dc:+.4f} points relative to coverage and {du:+.4f} relative to "
        +f"uniform attenuation with matched positive and negative element-space edit mass. "
        +"The supplementary local-edit study reports every task, both GT conventions and the "
        +"coverage-on-anchor-domain control. Other benchmark results remain unchanged references; "
        +"this is not an all-task transfer result.\n")
    (PAPER/'sections/nvos_local_credit_summary.tex').write_text(text)
    source=record['source_huber_objectives']
    lower={arm:sum(r[arm]<r['v10'] for r in source.values()) for arm in ARMS[1:]}
    lines=[f"The predeclared held-local arm obtains {100*primary['iou']:.4f}\\% IoU and "
           +f"{100*primary['average_precision']:.4f}\\% AP on retained GT."]
    for base,label in [('v10','the original v10'),('coverage','coverage'),('held_uniform','matched uniform attenuation')]:
        e=effects['retained_gt_gt127']['held_local_vs_'+base]['iou'];lo,hi=e['descriptive_scene_bootstrap95_pp']
        lines.append(f"Relative to {label}, its mean IoU change is {e['delta_pp']:+.4f} points "
            +f"({e['improved']} improved, {e['harmed']} harmed), with a descriptive seven-scene "
            +f"bootstrap interval [{lo:+.4f}, {hi:+.4f}].")
    lines.append(f"On the final fields, the common-source Huber objective is below the anchor "
        +f"for {lower['held_local']}/8 held-local and {lower['held_uniform']}/8 matched-uniform outputs. "
        +"These fields use all source constraints, so this objective is not an independent validation score. "
        +"The teacher-source criterion and target quality must be assessed separately.")
    (PAPER/'sections/nvos_local_credit_results.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=record['macro'],effects=effects),indent=2))


if __name__=='__main__':main()
