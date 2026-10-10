"""All LERF scene/query-view selection pairs, no selected success examples."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.run_query_field_compositional_atoms import ROOT,read

cfg=read(ROOT/'prediction_config.json');read(ROOT/'scoring_complete.json')
fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
for column,job in enumerate(cfg['jobs']):
    metrics=read(ROOT/'predictions'/job['name']/'evaluation/evaluation.json')['metrics']
    baseline=metrics['admission']['selected']['objects']
    for arm,color in [('teacher_union_protected','#4477aa'),('atom_protected','#cc6677')]:
        other=metrics[arm]['selected']['objects']
        assert [(x['frame'],x['query']) for x in baseline]==[(x['frame'],x['query']) for x in other]
        axes[0,column].scatter([x['iou'] for x in baseline],[x['iou'] for x in other],s=15,alpha=.55,c=color,label=arm)
    ax=axes[0,column];ax.plot([0,1],[0,1],c='gray',lw=1);ax.set(xlim=(0,1),ylim=(0,1),title=job['scene'],xlabel='Baseline actual-set IoU',ylabel='Candidate actual-set IoU');ax.set_aspect('equal')
    names=['admission','teacher_protected','teacher_union_protected','atom_parts','atom_protected'];positions=range(5)
    ax=axes[1,column];ax.bar([x-.18 for x in positions],[metrics[a]['iou']*100 for a in names],width=.36,label='Response')
    ax.bar([x+.18 for x in positions],[metrics[a]['selected']['miou']*100 for a in names],width=.36,label='Actual set')
    ax.set_xticks(list(positions),['Baseline','Teacher +P','Marginal union','Atoms','Atoms +P'],rotation=35);ax.set(ylim=(0,100),ylabel='IoU (%)')
axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
fig.suptitle('Source-assisted diagnostic: union before lifting with fixed teacher acceptance\nDots are all query-view evaluations, not independent scene samples')
out=ROOT/'diagnostics';out.mkdir(exist_ok=True)
for suffix in ['png','pdf']:fig.savefig(out/('all_scene_comparison.'+suffix),dpi=170)
print(out/'all_scene_comparison.png')
