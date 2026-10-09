"""Scientific diagnostic figure; all values use the same coarse evaluator."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.research.run_query_field_relations import read
from scripts.research.diagnose_query_field_region_composition import ROOT
r=read(ROOT/'summary.json');names=list(r['per_case']);rows=[r['per_case'][n] for n in names]
x=np.arange(len(names));fig,ax=plt.subplots(figsize=(11,4.8))
a=ax.bar(x-.19,[100*v['single_region_oracle'] for v in rows],.38,label='GT-best single region',color='#5e81ac')
b=ax.bar(x+.19,[100*v['actual']['iou'] for v in rows],.38,label='GT-assisted greedy composition, actual subset',color='#d08770')
ax.bar_label(a,fmt='%.1f',fontsize=8,padding=2);ax.bar_label(b,fmt='%.1f',fontsize=8,padding=2)
ax.set_xticks(x,names,rotation=20,ha='right');ax.set_ylim(0,110);ax.set_ylabel('Diagnostic IoU (%) at 192 x 256')
ax.set_title('Existing-region composition feasibility\nLabel-assisted diagnosis; NOT query-only accuracy or a global optimum')
ax.legend(loc='lower right',fontsize=8);ax.spines[['top','right']].set_visible(False)
fig.tight_layout();out=ROOT/'diagnostics';fig.savefig(out/'composition_capacity.png',dpi=180);fig.savefig(out/'composition_capacity.pdf')
