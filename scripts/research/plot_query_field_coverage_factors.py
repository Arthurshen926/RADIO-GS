"""Construction-function comparison; never label this a benchmark result."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.compile_query_field_coverage_factors import ROOT


def main():
    audit=json.loads(Path('docs/experiments/2026-10-10-coverage-factors-audit.json').read_text())
    names=['empty','full','random15','random50','back_half','x_half','soft']
    labels=['Empty','Full','Random 15%','Random 50%','Back half','X half','Soft']
    errors={k:0. for k in names}
    for scene in audit['scenes']:
        d=json.loads((ROOT/scene/'compilation.json').read_text())
        for view in d['records']:
            for c in view['cases']:errors[c['condition']]=max(errors[c['condition']],c['max_alpha_error'])
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),constrained_layout=True)
    baseline=audit['fixed_visibility_diagnostic'];x=list(range(len(names)))
    axes[0].plot(x,[max(1e-9,baseline[k]['max_error']) for k in names],'o-',label='Fixed full-set visibility')
    axes[0].plot(x,[max(1e-9,errors[k]) for k in names],'o-',label='Compiled joint opacity')
    axes[0].axhline(1e-4,color='grey',linestyle=':',label='Frozen acceptance tolerance')
    axes[0].set_yscale('log');axes[0].set_ylabel('Maximum absolute alpha error (zeros shown at 1e-9)');axes[0].legend(fontsize=8)
    axes[1].bar(x,[100*baseline[k]['selection_disagreements']/baseline[k]['samples'] for k in names],label='Fixed full-set visibility')
    axes[1].plot(x,[0]*len(names),'o-',color='tab:orange',label='Compiled joint opacity: zero disagreements')
    axes[1].set_ylabel('Silhouette decision disagreement (%)');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x,labels,rotation=25);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Sampled construction-function validation: 4 scenes / 72 views / 18,432 factors\nNo benchmark queries, no new segmentation accuracy result',fontsize=12)
    out=ROOT/'diagnostics';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'joint_vs_fixed_visibility.{ext}',dpi=160)

if __name__=='__main__':main()
