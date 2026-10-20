"""Execute one declared system recipe and its unchanged benchmark evaluators."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--recipe',default='configs/pipeline/system_v10.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--gpus',default='0,1,2,3,4,5')
    parser.add_argument('--benchmarks',nargs='+',help='Explicit subset of a declared recipe')
    parser.add_argument('--evidence-core',choices=['legacy','shared_v1'],
                        default=os.environ.get('RADIO_GS_EVIDENCE_CORE','legacy'))
    parser.add_argument('--scannet-paper8',action='store_true',
                        help='Also score the frozen paper-eight Gaussian metric on fresh outputs')
    args=parser.parse_args()
    recipe=json.loads(Path(args.recipe).read_text())
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    gpus=args.gpus.split(',')
    jobs=[j for j in recipe['jobs'] if not args.benchmarks or j['benchmark'] in args.benchmarks]
    paper8=[]
    if args.scannet_paper8:
        import yaml
        freeze=yaml.safe_load(Path('paper/artifacts/evaluation_protocol_freeze_20260801.yaml').read_text())
        paper8=freeze['canonical_tasks']['concept_scannet_ovs_vala_paper8']['cohort']['scenes']
        supplied={j['name'] for j in jobs if j['benchmark']=='scannet'}
        if not set(paper8)<=supplied:
            raise ValueError('paper-eight scoring requires the complete frozen cohort')

    def predict(index):
        results=[]
        for job in jobs[index::len(gpus)]:
            started=time.monotonic()
            out=root/job['benchmark']/job['name'];out.mkdir(parents=True,exist_ok=True)
            command=[PYTHON,'-m','radio_gs.pipeline','--scene',job['scene'],
                     '--queries',job['queries'],'--access',job.get('access','source-assisted'),
                     '--output',str(out/'query'),'--gpu',gpus[index]]
            if job.get('source_evidence'):
                command.extend(['--source-evidence',job['source_evidence']])
            if job.get('reference_tracks'):
                command.extend(['--reference-tracks',job['reference_tracks']])
            if job.get('reference_completion'):
                command.extend(['--reference-completion',job['reference_completion']])
            for option in ['scope_transfer','instance_scope']:
                if job.get(option):command.append('--'+option.replace('_','-'))
            env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[index],OPENBLAS_NUM_THREADS='1',
                     RADIO_GS_EVIDENCE_CORE=args.evidence_core,
                     RADIO_GS_EVIDENCE_TRACE_DIR=str((out/'core_trace').resolve()))
            with (out/'query.log').open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            results.append(dict(job=job,commands=[command],evidence_core=args.evidence_core,
                                query_seconds=time.monotonic()-started))
            print('predicted',job['benchmark'],job['name'],flush=True)
        return results

    with ThreadPoolExecutor(len(gpus)) as pool:
        predicted=[r for group in pool.map(predict,range(len(gpus))) for r in group]
    (root/'predictions_complete.json').write_text(json.dumps(dict(recipe=args.recipe,jobs=len(predicted),
        evidence_core=args.evidence_core,scannet_paper8=paper8,
        all_predictions_before_scoring=True,historical_answers_read=False),indent=2)+'\n')
    from radio_gs.pipeline.prediction_receipts import prediction_seals
    (root/'prediction_seal.json').write_text(json.dumps(dict(
        outputs=prediction_seals(root,dict(jobs=jobs)),sealed_before_scoring=True),indent=2)+'\n')

    public=[r for r in predicted if r['job']['benchmark']!='uqis']
    def evaluate(index):
        results=[]
        for result in public[index::len(gpus)]:
            job=result['job'];out=root/job['benchmark']/job['name']
            env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[index],OPENBLAS_NUM_THREADS='1')
            if job['benchmark']=='lerf':
                evaluate=[PYTHON,'scripts/research/evaluate_consolidated_query.py','--scene',job['scene'],
                          '--readout',str(out/'query/typed_readout.pt'),'--output',str(out/'evaluation')]
            elif job['benchmark']=='nvos':
                evaluate=[PYTHON,'scripts/research/evaluate_reference_instance.py','--scene',job['scene'],
                          '--evaluation',job['evaluation'],'--field',str(out/'query/typed_readout.pt'),
                          '--typed-readout','--output',str(out/'evaluation')]
            else:
                evaluate=[PYTHON,'scripts/research/evaluate_scannet_runtime.py',
                          '--readout',str(out/'query/category/typed_readout.pt'),
                          '--evaluation',job['evaluation'],'--output',str(out/'evaluation')]
                if job['name'] not in paper8:evaluate.append('--point-only')
            with (out/'evaluation.log').open('w') as log:
                subprocess.run(evaluate,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            if job['benchmark']=='lerf':
                d=json.loads((out/'evaluation/evaluation.json').read_text())['metrics']['hierarchical_pipeline']
                metrics=dict(semantic_iou=d['iou'],ap=d['ap'],selected_projection_iou=d['selected']['miou'])
            elif job['benchmark']=='nvos':
                d=json.loads((out/'evaluation/evaluation.json').read_text())['metrics']
                metrics=dict(iou=d['iou'],ap=d['average_precision'])
            else:
                d=json.loads((out/'evaluation/report.json').read_text())['point_metrics']['source']
                metrics={f'miou_{s}':d[s]['miou'] for s in ['19','15','10']}
            result={**result,'commands':result['commands']+[evaluate],'metrics':metrics}
            (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            results.append(result)
            print(job['benchmark'],job['name'],metrics,flush=True)
        return results

    with ThreadPoolExecutor(len(gpus)) as pool:
        rows=[r for group in pool.map(evaluate,range(len(gpus))) for r in group]
    uqis=[r for r in predicted if r['job']['benchmark']=='uqis']
    uqis_readouts={}
    if uqis:
        uqis_root=root/'uqis';first=uqis[0]['job']
        for readout in first['readouts']:
            folder=uqis_root/readout;predictions=folder/'predictions';predictions.mkdir(parents=True,exist_ok=True)
            for row in uqis:
                source=uqis_root/row['job']['name']/'query/registered'/readout/'predictions'
                for file in source.glob('*.npy'):shutil.copyfile(file,predictions/file.name)
                selected=source.parent/'selections'
                if selected.exists():
                    destination=folder/'selections';destination.mkdir(parents=True,exist_ok=True)
                    for file in selected.glob('*.npy'):shutil.copyfile(file,destination/file.name)
            command=[PYTHON,'-m','radio_gs.benchmarks.uqis.evaluate','--package',first['package'],
                '--predictions',str(predictions),'--output',str(folder/'result.json')]
            if (folder/'selections').exists():command.extend(['--selections',str(folder/'selections')])
            with (folder/'evaluation.log').open('w') as log:
                subprocess.run(command,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
            d=json.loads((folder/'result.json').read_text());uqis_readouts[readout]=d
        for row in uqis:
            job=row['job'];d=uqis_readouts[job['primary_readout']]
            metrics={m+'_ap':v['per_scene'][job['name']]['average_precision'] for m,v in d['modalities'].items()}
            if 'explicit_selection' in d:
                metrics.update({m+'_selected_iou':v['per_scene'][job['name']]['iou'] for m,v in d['explicit_selection']['modalities'].items()})
            result={**row,'metrics':metrics,'evaluation':str(uqis_root/job['primary_readout']/'result.json')}
            (uqis_root/job['name']/'result.json').write_text(json.dumps(result,indent=2)+'\n');rows.append(result)
    macro={}
    for name in dict.fromkeys(row['job']['benchmark'] for row in rows):
        group=[r['metrics'] for r in rows if r['job']['benchmark']==name]
        macro[name]={k:sum(r[k] for r in group)/len(group) for k in group[0]}
    summary=dict(recipe=args.recipe,completed=len(rows),rows=rows,macro=macro,
                 evidence_core=args.evidence_core,
                 source_teacher_caches_reused=True,historical_query_predictions_read=False,
                 all_predictions_before_scoring=True,
                 uqis_readouts={n:{m:v['scene_macro'] for m,v in d['modalities'].items()} for n,d in uqis_readouts.items()},
                 uqis_selections={n:{m:v['scene_macro'] for m,v in d['explicit_selection']['modalities'].items()}
                                  for n,d in uqis_readouts.items() if 'explicit_selection' in d},
                 independent_test=False,official_sota_equivalence_verified=False)
    if paper8:
        gaussian={name:json.loads((root/'scannet'/name/'evaluation/report.json').read_text())[
            'gaussian_volume_pseudolabel_metrics']['source'] for name in paper8}
        summary['scannet_paper8']=dict(scenes=paper8,
            macro={split:{metric:sum(gaussian[n][split][metric] for n in paper8)/len(paper8)
                for metric in ['miou','macc']} for split in ['19','15','10']},
            metric='Mahalanobis pseudo labels; opacity-volume weighting; equal scene mean',
            input_track='provided unlabelled mesh; source-assisted teachers',
            exact_vala_input_equivalence=False,shared_fresh_predictions_with_internal10=True)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(macro,indent=2),flush=True)


if __name__=='__main__':
    main()
