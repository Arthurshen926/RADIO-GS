"""Render fixed-cohort attribution; coverage is explicitly not model accuracy."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('output/query_field_20261009/extent_stages_v1')
r=json.loads((ROOT/'summary.json').read_text());names=list(r['per_case']['local']);x=np.arange(len(names))
fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
for arm,label in [('all_regions','All-region optimistic recall'),('compatible_regions','Scale-compatible optimistic recall'),('admitted_regions','Admitted-region optimistic recall'),('local','Actual selected-set recall')]:
 axes[0].plot(x,[100*r['per_case'][arm][s]['recall'] for s in names],marker='o',label=label)
axes[0].set(ylabel='Target recall (%)',ylim=(0,105),title='Coverage does not establish instance discrimination');axes[0].legend(ncol=2,fontsize=9)
for i,arm in enumerate(['no_region','no_negative','unary']):
 axes[1].bar(x+(i-1)*.24,[100*(r['per_case'][arm][s]['selection_iou']-r['per_case']['local'][s]['selection_iou']) for s in names],width=.24,label=arm)
axes[1].axhline(0,color='black',linewidth=.7);axes[1].set(ylabel='Actual IoU change (pp)',title='Removing region evidence or negative conditioning is not a consistent repair');axes[1].legend()
for ax in axes:ax.set_xticks(x,names);ax.grid(axis='y',alpha=.2)
out=ROOT/'diagnostics';out.mkdir(exist_ok=True)
for suffix in ['png','pdf']:fig.savefig(out/('extent_stages.'+suffix),dpi=160)
