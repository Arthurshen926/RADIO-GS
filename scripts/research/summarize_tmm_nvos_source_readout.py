"""Generate N02 tables from a complete source-calibration audit only."""
import csv
import json
from pathlib import Path
from statistics import mean

from radio_gs.response_study.source_readout_study import check_source_readout_study, check_score_changes
from radio_gs.response_study.reference_study import read

REPO=Path(__file__).resolve().parents[2]
PAPER=REPO/'paper/tmm_response'


def plot_source_criteria(record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,4,figsize=(11.5,5.4),sharex=True,sharey=True)
    for ax,(name,arms) in zip(axes.flat,record['source_calibration'].items()):
        for arm,color,label in [('source_fit','#2864a0','Fit-source'),('source_holdout','#b65820','Held-source')]:
            r=arms[arm];p=np.asarray(r['positive_histogram']);n=np.asarray(r['negative_histogram'])
            tp=np.cumsum(p[::-1])[::-1];fp=np.cumsum(n[::-1])[::-1]
            score=tp/np.maximum(p.sum()+fp,1e-12);x=np.arange(r['bins'])/r['bins']
            ax.plot(x[1:],score[1:],color=color,label=label,lw=1.1)
            ax.scatter([r['threshold']],[r['soft_iou']],color=color,s=18,zorder=3)
        ax.axvline(.5,color='#777777',lw=.8,linestyle='--');ax.set_title(name.replace('_',' '),fontsize=10)
        ax.grid(alpha=.18);ax.set_xlim(0,1);ax.set_ylim(0,1)
    for ax in axes[-1]:ax.set_xlabel('Source-estimated threshold')
    for ax in axes[:,0]:ax.set_ylabel('Soft teacher-source IoU')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
    fig.tight_layout(rect=[0,0,1,.94])
    for ext in ('png','pdf'):
        fig.savefig(PAPER/f'figures/nvos_source_criteria.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def main():
    registry=read(REPO/'configs/research/tmm_response.json')
    item=next(r for r in registry['reference_studies'] if r['id']=='N02')
    record=check_source_readout_study(REPO,REPO/registry['study_root'],item,read(REPO/registry['recipe'])['jobs'])
    if not record['complete']:raise ValueError('N02 has not completed all eight tasks')
    import numpy as np
    effects={};rows=record['rows'];flat=[]
    for track in record['macro']:
        effects[track]={}
        for arm in ('source_fit','source_holdout'):
            delta={n:100*(r[arm][track]['iou']-r['fixed_half'][track]['iou']) for n,r in rows.items()}
            clusters={n:('horns' if n.startswith('horns_') else n) for n in rows}
            unique=sorted(set(clusters.values()));rng=np.random.default_rng(0);boot=[]
            for sampled in rng.choice(unique,size=(5000,len(unique)),replace=True):
                boot.append(mean(d for c in sampled for n,d in delta.items() if clusters[n]==c))
            effects[track][arm]=dict(delta_pp=mean(delta.values()),per_task_delta_pp=delta,
                improved=sum(v>1e-6 for v in delta.values()),harmed=sum(v< -1e-6 for v in delta.values()),
                unchanged=sum(abs(v)<=1e-6 for v in delta.values()),independent_scene_clusters=7,
                descriptive_scene_bootstrap95_pp=np.quantile(boot,[.025,.975]).tolist())
    record['effects']=effects
    record['source_criterion_improvements']={n:{a:r['soft_iou']-r['soft_iou_at_half'] for a,r in arms.items()}
                                            for n,arms in record['source_calibration'].items()}
    if (REPO/registry['study_root']/'nvos_score_changes_v1/summary.json').exists():
        attribution=check_score_changes(REPO/registry['study_root'])
        record['score_change_attribution']=attribution['rows']
        record['receipts'].update(attribution['receipts'])
        lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
               r'Task & v10 AP & Coverage AP & Foreground $\Delta$ & Background $\Delta$ \\',r'\midrule']
        for name,row in attribution['rows'].items():
            a=row['average_precision']
            lines.append(name.replace('_',r'\_')+f" & {100*a['v10']:.2f} & {100*a['coverage']:.2f}"
                         +f" & {a['foreground_change_pp']:+.3f} & {a['background_change_pp']:+.3f}"+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}']
        (PAPER/'tables/nvos_score_changes.tex').write_text('\n'.join(lines)+'\n')
    for n,arms in rows.items():
        for a,tracks in arms.items():
            for t,metrics in tracks.items():
                flat.append(dict(task=n,arm=a,gt_track=t,threshold=record['thresholds'][n][a],**metrics))
    (PAPER/'nvos_source_readout_feedback.json').write_text(json.dumps(record,indent=2)+'\n')
    with (PAPER/'artifacts/nvos_source_readout.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    arms=['fixed_half','source_fit','source_holdout']
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',
           r'Task & Fixed 0.5 & Fit-source & Held-source & $t_{\rm fit}$ & $t_{\rm hold}$ \\',r'\midrule']
    for n,values in rows.items():
        lines.append(n.replace('_',r'\_')+' & '+' & '.join(f"{100*values[a]['retained_gt_gt127']['iou']:.2f}" for a in arms)
                     +' & '+' & '.join(f"{record['thresholds'][n][a]:.4f}" for a in arms[1:])+r' \\')
    for track,label in [('retained_gt_gt127','Mean (retained GT)'),('ludvig_gt_maximum','Mean (LUDVIG GT)')]:
        lines.append(r'\midrule '+label+' & '+' & '.join(f"{100*record['macro'][track][a]['iou']:.2f}" for a in arms)+r' & -- & -- \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/nvos_source_readout.tex').write_text('\n'.join(lines)+'\n')
    plot_source_criteria(record)
    print(json.dumps(dict(macro=record['macro'],effects=effects,thresholds=record['thresholds']),indent=2))


if __name__=='__main__':main()
