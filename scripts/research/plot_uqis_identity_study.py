"""Plot actual full-cohort retrieval; label-assisted capacity stays separate."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,default=Path('output/observation_hypotheses_study/comparison.json'))
    p.add_argument('--output',type=Path,default=Path('output/observation_hypotheses_study/comparison'))
    a=p.parse_args();results=json.loads(a.comparison.read_text())['results']
    names={'v7':'v7 baseline','scope_hypotheses':'Multi-view mean','view_exclusive_tracks':'Exclusive tracks',
        'native_local_appearance':'DINO appearance','dual_native_identity':'Dual identity ranks'}
    modalities=['text','image','point_2d','point_3d'];x=np.arange(4)
    colors=['#234a67','#b7bbc0','#ce9086','#64a398','#d09b3a']
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained');width=.8/len(results)
    for axis,metric,title in zip(axes,['ap','iou'],['Ranking AP','Actual selected-set IoU']):
        for index,(name,result) in enumerate(results.items()):
            values=[result['modalities'][m][metric] for m in modalities]
            axis.bar(x+(index-(len(results)-1)/2)*width,values,width,color=colors[index],label=names[name])
        axis.set_xticks(x,['Text','Image','2D point','3D point']);axis.set_title(title)
        axis.set_ylabel('Scene macro (%)');axis.set_ylim(0,65);axis.grid(axis='y',alpha=.2)
        axis.set_axisbelow(True);axis.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=len(results),frameon=False)
    fig.suptitle('UQIS development: 9 scenes, 67 targets, all 268 independent queries',fontsize=12)
    for extension in ['png','pdf']:fig.savefig(a.output.with_suffix('.'+extension),dpi=180,bbox_inches='tight')


if __name__=='__main__':main()
