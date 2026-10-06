"""Report frozen local-fusion experiments without selecting a policy."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('output/local_element_fusion')
MODES=['siglip_vote','adapter_vote','average_prior','local_single','local_dual','global_dual']
LABELS=['SigLIP vote','Adapter vote','Equal prior average','Local / single','Local / dual','Global / dual']

def main():
    cohorts={key:json.loads((path/'summary.json').read_text()) for key,path in
             [('development',ROOT/'development'),('reserved',ROOT)]}
    train={name:json.loads((ROOT/f'{name}_fit.json').read_text()) for name in MODES[3:]}
    report={'training_only':{},'cohorts':{},'learned_readout_promoted':False}
    for mode,data in train.items():
        report['training_only'][mode]={sp:100*np.mean([row[sp] for row in data['scenes'].values()]) for sp in ['19','15','10']}
    for cohort,data in cohorts.items():
        modes={mode:{sp:100*data['macro'][mode]['source'][sp] for sp in ['19','15','10']} for mode in MODES}
        baseline=data['scenes']['siglip_vote'];delta={}
        for mode in MODES:
            delta[mode]={name:100*(row['source']['19']['miou']-baseline[name]['source']['19']['miou']) for name,row in data['scenes'][mode].items()}
        report['cohorts'][cohort]={'macro_percent':modes,'scene_delta_pp19':delta,'scenes':list(baseline),
            'new_scene_for_this_mechanism':data['new_scene_for_this_mechanism']}
    timing={}
    for mode in MODES:
        timing[mode]={name:json.loads((ROOT/'evaluation'/mode/name/'query/result.json').read_text())['seconds']
                     for name in cohorts['reserved']['scenes'][mode]}
    report['public_query_seconds']=timing
    report['checkpoint_bytes']={mode:(ROOT/f'{mode}.pt').stat().st_size for mode in MODES[3:]}
    storage={}
    for name in cohorts['reserved']['scenes']['siglip_vote']:
        base=ROOT/'maps'/name
        s=json.loads((base/'compiled/scene.json').read_text());a=json.loads((base/'adapter/mask_adapter/scene.json').read_text())
        storage[name]={key:Path(value).stat().st_size for key,value in
            [('siglip_dictionary_and_transfer',s['native_observation_field']),('adapter_dictionary_and_transfer',a['native_observation_field'])]}
        storage[name]['source_detector_identity_and_projection']=sum(p.stat().st_size for p in (ROOT/'evidence'/name).rglob('*.pt'))
    report['additional_asset_bytes']=storage
    report['cost_excludes']='RGB, geometry, pretrained encoders, SAM and build-time caches; two dictionaries currently duplicate sparse transfer storage'
    report['promotion_record']='configs/pipeline/system_v6.json; equal prior only, learned heads remain research'
    (ROOT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),constrained_layout=True)
    x=np.arange(len(MODES));colors=['#777777','#aaaaaa','#e6a23c','#54a2b6','#286c9b','#9c6bad']
    for i,(cohort,title) in enumerate([('development','Previously opened development (6 rooms)'),('reserved','Reserved evaluation (3 new rooms)')]):
        values=[report['cohorts'][cohort]['macro_percent'][mode]['19'] for mode in MODES]
        axes[i].bar(x,values,color=colors)
        axes[i].set_xticks(x,LABELS,rotation=35,ha='right');axes[i].set_ylim(0,100)
        axes[i].set_ylabel('Scene-macro point mIoU19 (%)');axes[i].set_title(title)
        axes[i].axhline(values[0],color='#777777',linestyle='--',linewidth=1)
        for j,value in enumerate(values):axes[i].text(j,value+1,f'{value:.2f}',ha='center',fontsize=9)
    fig.suptitle('Frozen geometry and native spaces; learned readers use 9 official-training rooms')
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'transfer_results.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps(report['cohorts'],indent=2))

if __name__=='__main__':main()
