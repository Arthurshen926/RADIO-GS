"""Preregister and run two paired source pilots, without benchmark access."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from radio_gs.utils.immutable_artifacts import sha256_file

BASE=Path('output/pair_ranked_scope_next32')
TRAINER='scripts/audits/train_pair_ranked_scope_next32.py'
HELPER='radio_gs/training/disjoint_query_competition.py'
PILOTS=[
    dict(key='lerf_figurines',root='output/full_field_rebuild_next19/figurines/reconstruction/field',
         geometry='output/depth_opacity_repair_20260918/figurines/geometry_report.json',guard='scripts/audits/run_lerf_source_stage.py'),
    dict(key='scannet_scene0000_00',root='output/scannet_expanded_budget_next29/scene0000_00/reconstruction/field',
         geometry='output/scannet_expanded_budget_next29/scene0000_00/geometry_report.json',guard='scripts/audits/run_scannet_source_stage.py'),
]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--pilot',choices=[p['key'] for p in PILOTS])
    args=parser.parse_args()
    if args.register:
        BASE.mkdir(exist_ok=False)
        inputs={}
        for row in PILOTS:
            root=Path(row['root'])
            for path in [root/'bounded_fallback_admission/field.pt',root/'teachers.json',root/'responsibility.json',
                         root/'caption_region_evidence.pt',Path(row['geometry'])]:
                inputs[str(path)]=sha256_file(path)
        reg=dict(pilots=PILOTS,modes=['visible_selected','rank_guarded'],steps=400,seed=42,lr=1e-4,
            source_only=True,benchmark_data_allowed=False,mainline_promoted=False,
            competition='Same native query coordinates, mask-disjoint scopes and reciprocal source crop/query preference; pixel zero-margin hinge only inside positive masks; weight1. No whole/part or unknown background negatives.',
            pairs='All eligible training pairs; per-step fixed source-order cycle in current training view. Pairs with either mask lacking valid observed support excluded and counted. Validation pairs first20 per view, derived independently from heldout source evidence before outcomes.',
            validation='Last4 source views excluded only from this head fit; parent saw them; not independent test.',
            checkpoint='Both arms identical: full IoU no decrease, selected IoU improves, pairaccuracy no decrease, pairhinge no increase. Stepzero eligible.',
            expansion_gate='Both pilots: rankarm chosen fullIoU,selectedIoU,pairaccuracy no worse than chosen control; pairhinge at least5% lower than chosencontrol and strictly lower. No target run otherwise.',
            implementations={p:sha256_file(p) for p in [__file__,TRAINER,HELPER,'radio_gs/training/selected_output_supervision.py']},
            inputs=inputs)
        (BASE/'registration.json').write_text(json.dumps(reg,indent=2)+'\n')
        print('registered',flush=True)
        return
    reg=json.loads((BASE/'registration.json').read_text())
    for p,d in reg['implementations'].items():assert sha256_file(p)==d,p
    assert args.pilot
    row=next(r for r in reg['pilots'] if r['key']==args.pilot)
    for p,d in reg['inputs'].items():
        if p.startswith(row['root']) or p==row['geometry']:assert sha256_file(p)==d,p
    out=BASE/args.pilot;out.mkdir(exist_ok=True)
    for mode in reg['modes']:
        dst=out/mode
        if dst.exists():
            report=json.loads((dst/'report.json').read_text())
            assert report['implementation_sha256']==reg['implementations'][TRAINER]
            assert sha256_file(dst/'field.pt')==report['field_sha256']
            continue
        command=[sys.executable,row['guard'],TRAINER,'--root',row['root'],'--geometry-report',row['geometry'],
                 '--mode',mode,'--steps','400','--parent-name','bounded_fallback_admission','--device','cuda:0','--output',str(dst)]
        with (out/(mode+'.log')).open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
        assert result.returncode==0,(mode,'Preserve failed attempt and inspect log')
        print(args.pilot,mode,'complete',flush=True)
    def chosen(mode):
        result=json.loads((out/mode/'report.json').read_text())
        return next(h['validation'] for h in result['history'] if h['step']==result['selected_step'])
    control=chosen('visible_selected');rank=chosen('rank_guarded')
    gate={k:rank[k]>=control[k]-1e-6 for k in ('full_iou','selected_iou','query_order_accuracy')}
    gate['pair_hinge_5pct_reduction']=rank['query_order_hinge']<=.95*control['query_order_hinge'] and rank['query_order_hinge']<control['query_order_hinge']-1e-8
    summary=dict(pilot=args.pilot,control=control,rank_guarded=rank,gate=gate,source_admitted=all(gate.values()),
                 source_only=True,independent_test=False,benchmark_evaluation_launched=False,
                 registration_sha256=sha256_file(BASE/'registration.json'),
                 report_hashes={mode:sha256_file(out/mode/'report.json') for mode in reg['modes']})
    with (out/'summary.json').open('x') as file:json.dump(summary,file,indent=2)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
