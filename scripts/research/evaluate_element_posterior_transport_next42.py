"""Six fixed Stage3 real evaluations; hash-bound Stage1 B0 is reused."""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import torch
from radio_gs.querying.query_region_evidence import load,digest

KEYS=['lerf_figurines','lerf_ramen','scannet_scene0000_00']
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'


def main(out):
    gate=json.loads((out/'source_gate.json').read_text())
    frozen={'recipe':digest(out/'recipe.json'),'source_gate':digest(out/'source_gate.json'),**{arm:digest(out/arm/'head.pt') for arm in ['B1','B2']}}
    (out/'frozen_before_target.json').write_text(json.dumps(frozen,indent=2))
    frozen['transport']={key:digest(out/key/'transport.pt') for key in KEYS}
    frozen['banks']={key:digest(out/key/'bank.pt') for key in KEYS}
    (out/'frozen_before_target.json').write_text(json.dumps(frozen,indent=2))
    execution=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONPATH='.')
    report=dict(complete=False,promotion=False,development=True,source_gate=gate,scenes={},frozen=frozen)
    stage1=Path(json.loads((out/'recipe.json').read_text())['stage1'])
    previous=json.loads((stage1/'report.json').read_text())
    for key in KEYS:
        row=dict(previous['scenes'][key]['B0'])
        row.update(reused_B0=True,evaluation_sha256=digest(row['evaluation']),predictions_sha256=digest(row['predictions']))
        report['scenes'][key]={'B0':row}
    for arm in ['B1','B2']:
        for key in KEYS:
            directory=out/arm/key;directory.mkdir(parents=True,exist_ok=True)
            parent=Path('output/taskwise_scope_composition_next38')/key/'field.pt'
            addon=dict(kind='element_posterior',transport=str(out/key/'transport.pt'),arm=arm,bank=str(out/key/'bank.pt'),checkpoint=str(out/arm/'head.pt'),parent_scores=str(parent.parent/'anchored/pipeline_calibrated/scores.pt'))
            torch.save(dict(next42_parent=str(parent),query_region_evidence=addon),directory/'field.pt')
            shutil.copyfile(parent.parent/'calibration.pt',directory/'calibration.pt')
            argv=[PYTHON,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(out/arm),'--readout','anchored','--pipeline','--calibrated']
            log=directory/'evaluation.log';entry=dict(arm=arm,key=key,argv=argv,log=str(log),started=time.time());execution.append(entry)
            (out/'execution.json').write_text(json.dumps(execution,indent=2));start=time.monotonic()
            with log.open('w') as f:entry['exit_code']=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
            entry['seconds']=time.monotonic()-start;(out/'execution.json').write_text(json.dumps(execution,indent=2))
            if entry['exit_code']:raise RuntimeError(str(entry))
            result=directory/'anchored/pipeline_calibrated';metrics=json.loads((result/'evaluation.json').read_text())['metrics']['hierarchical_pipeline']
            row=dict(field_addition=str(directory/'field.pt'),evaluation=str(result/'evaluation.json'),predictions=str(result/'scores.pt'),metrics=metrics,parent_sha256=digest(parent),diagnostic_only=(arm!=gate['preferred_arm']))
            if arm=='B0':
                historical=parent.parent/'anchored/pipeline_calibrated'
                current=load(result/'scores.pt');old=load(historical/'scores.pt')
                row['exact_next38_replay']=all(torch.equal(current['scores'][k],old['scores'][k]) for k in old['scores'])
                def comparable(x):
                    if isinstance(x,dict):return {k:comparable(v) for k,v in x.items() if k!='pred_path'}
                    if isinstance(x,list):return [comparable(v) for v in x]
                    return x
                row['metric_replay']=comparable(metrics)==comparable(json.loads((historical/'evaluation.json').read_text())['metrics']['hierarchical_pipeline'])
                assert row['exact_next38_replay'], 'frozen parent scores are not exact'
            report['scenes'].setdefault(key,{})[arm]=row
            (out/'report.json').write_text(json.dumps(report,indent=2));print(key,arm,'completed',flush=True)
    report['complete']=True
    for arm in ['B1','B2']:
        lerf={m:sum((report['scenes'][k][arm]['metrics'][m] if m!='selected' else report['scenes'][k][arm]['metrics']['selected']['miou'])-(report['scenes'][k]['B0']['metrics'][m] if m!='selected' else report['scenes'][k]['B0']['metrics']['selected']['miou']) for k in KEYS[:2])/2 for m in ['iou','ap','selected']}
        # All existing scorer metrics are fractions.
        lerf={k:100*v for k,v in lerf.items()}
        scan={split:100*(report['scenes'][KEYS[2]][arm]['metrics'][split]['miou']-report['scenes'][KEYS[2]]['B0']['metrics'][split]['miou']) for split in ['19','15','10']}
        report.setdefault('target_screen',{})[arm]=dict(lerf_delta_pp=lerf,scannet_delta_pp=scan,passed=lerf['iou']>=.5 and lerf['selected']>=.5 and lerf['ap']>=-.25 and min(scan.values())>=-.5)
    assert digest(out/'recipe.json')==frozen['recipe'] and digest(out/'source_gate.json')==frozen['source_gate']
    for arm in ['B1','B2']:assert digest(out/arm/'head.pt')==frozen[arm]
    (out/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(Path(a.output))
