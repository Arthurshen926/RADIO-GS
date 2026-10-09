"""All-source semantic union controls on the complete four-scene LERF cohort."""
import argparse
import copy
import gc
from pathlib import Path
import subprocess
import torch

from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
from radio_gs.query_field.storage import load_field

ROOT=Path('output/query_field_20261009/text_union_functions_v1')
PREVIOUS=Path('output/query_field_20261009/confidence_mixture_functions_v1')
PROTOCOL=Path('docs/experiments/2026-10-09-text-union-functions-protocol.md')
CONTROLS=('mean_prior','mean_typed','measure_prior','measure_typed')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);c=read(PREVIOUS/'confidence_mixture_function/cohort.json')
    previous=read(PREVIOUS/'design.json');old=read(Path(c['lerf'][0]['predictions'])/'prediction.json')['configuration']
    config=copy.deepcopy(old);config['jobs']=[];config['status']='fixed full-source independent union study;no promotion'
    config['selection']='each control own >.5 set;same retained alpha-byte10 scorer'
    jobs=[]
    for item in c['lerf']:
        name=item['scene'];template=next(j for j in old['jobs'] if j['scene']==name)
        obs=next(j['observations'] for j in previous['fields'] if j['family']=='lerf' and j['scene']==name)
        source=read(Path(obs)/'observations.json');scene=read(template['evaluation_scene'])
        frames=[f for f in scene['source_frames'] if f not in source['splits']['development_frames']]
        jobs.append(dict(name=name,parent=item['field'],source_scene=template['evaluation_scene'],queries=template['queries'],
            observations=obs,frames=frames,excluded_frames=source['splits']['development_frames'],
            receipts=[receipt(p) for p in [Path(item['field'])/'field.pt',Path(item['field'])/'manifest.json',
                template['evaluation_scene'],template['queries'],Path(obs)/'observations.json',scene['source_rgb_authority']]]))
        for arm in ['parent','measure']:
            config['jobs'].append(dict(template,name=name+'__'+arm,field=str(ROOT/'fits'/name/'field')))
    if len(jobs)!=4 or sum(len(j['frames']) for j in jobs)!=756:raise ValueError('complete legal LERF roster differs')
    write(ROOT/'prediction_config.json',config)
    write(ROOT/'design.json',dict(jobs=jobs,configuration=receipt(ROOT/'prediction_config.json'),
        protocol=receipt(PROTOCOL),code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(Path(__file__))],
        primary='measure_typed',controls=CONTROLS,training_steps=0,new_member_supervision=False,
        v10_source_query_calibration=False,whole_five_task=False,model_promotion=False,
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d['configuration'],d['protocol']]:check(r)
    return d


@torch.no_grad()
def predict_scene(j,d):
    from radio_gs.query_field.adapters import NativeQueryEncoder
    from radio_gs.query_field.contracts import QueryPacket
    from radio_gs.querying.observation_readout import positive_support_rank
    config=read(d['configuration']['path']);queries=read(j['queries'])['queries'];texts=[q['text'] for q in queries]
    encoder=NativeQueryEncoder(config['shared_encoder'],'cuda')
    vectors=torch.stack([encoder.text(t).semantic.cpu().float() for t in texts])
    negatives=torch.stack([encoder.text(t).semantic.cpu().float() for t in config['generic_negative_texts']])
    del encoder;gc.collect();torch.cuda.empty_cache()
    field,manifest=load_field(ROOT/'fits'/j['name']/'field','cuda')
    background=field.semantic.score(negatives.cuda()).max(1).values
    native=[];member=[];selected=[];scores={k:[] for k in CONTROLS};sets={k:[] for k in CONTROLS};diagnostics=[]
    available=field.observed|field.semantic_measure.supported
    for text,vector in zip(texts,vectors):
        result=field.query(QueryPacket('text',task='union',semantic=vector.cuda()))
        old=(config['native_temperature']*(result.semantic_scores-background)).sigmoid()
        native.append(old.cpu());member.append(result.membership_scores.cpu());selected.append(result.selected.cpu())
        row=dict(text=text,parent_selected=int(result.selected.sum()),member_candidates=len(result.candidate_indices),controls={})
        for rule in ['mean','measure']:
            prior=field.semantic_measure.union_probabilities(vector.cuda(),negatives.cuda(),old,rule,config['native_temperature'])
            for suffix,value in [('prior',prior),('typed',positive_support_rank(prior,result.membership_scores))]:
                key=rule+'_'+suffix;value=value.masked_fill(~available,0);subset=value>field.threshold
                scores[key].append(value.cpu());sets[key].append(subset.cpu())
                row['controls'][key]=dict(selected=int(subset.sum()),semantic_modes=len(field.semantic_measure.descriptors))
        diagnostics.append(row);print('predicted',j['name'],text,flush=True)
    geometry={k:getattr(field,k).cpu() for k in ['xyz','normals','scales','quaternions','opacities']}
    for arm in ['parent','measure']:
        job=next(x for x in config['jobs'] if x['name']==j['name']+'__'+arm);out=ROOT/'predictions'/job['name'];out.mkdir(parents=True,exist_ok=False)
        payload=dict(texts=texts,geometry=geometry,observed=field.observed.cpu() if arm=='parent' else available.cpu(),
            native_query_vectors=vectors,generic_negative_vectors=negatives)
        if arm=='parent':payload.update(scores=dict(native_generic_relevance=torch.stack(native,1),shared_membership=torch.stack(member,1)),selected=torch.stack(selected,1))
        else:payload.update(scores={k:torch.stack(v,1) for k,v in scores.items()},selected_by_control={k:torch.stack(v,1) for k,v in sets.items()},selected=torch.stack(sets['measure_typed'],1))
        torch.save(payload,out/'predictions.pt')
        write(out/'prediction.json',dict(job=job,configuration=config,queries=len(texts),diagnostics=diagnostics,
            predictions_sha256=receipt(out/'predictions.pt')['sha256'],field=manifest['files'],queries_receipt=receipt(j['queries']),
            runtime_source_access=False,target_labels_opened=False,evaluation_scene_opened_by_predictor=False,
            legacy_calibration_reused=False,primary='measure_typed'))


def verify_predictions(d):
    config=read(d['configuration']['path']);records=[]
    for j in config['jobs']:
        path=ROOT/'predictions'/j['name'];r=read(path/'prediction.json')
        if r['configuration']!=config:raise ValueError('query protocol changed')
        check(dict(path=str(path/'predictions.pt'),sha256=r['predictions_sha256']));records.append(receipt(path/'prediction.json'))
    return records


def run(d,gpus):
    def build(j,gpu):
        execute(['-m','scripts.research.run_query_field_text_union','compile','--name',j['name']],ROOT/'logs/build'/(j['name']+'.log'),gpu)
    lanes(d['jobs'],gpus,build)
    write(ROOT/'construction_complete.json',dict(records=[receipt(ROOT/'fits'/j['name']/'construction.json') for j in d['jobs']]))
    def predict(j,gpu):execute(['-m','scripts.research.run_query_field_text_union','predict','--name',j['name']],ROOT/'logs/predict'/(j['name']+'.log'),gpu)
    lanes(d['jobs'],gpus,predict)
    write(ROOT/'predictions_complete.json',dict(records=verify_predictions(d)))
    score(d,gpus)


def score(d,gpus):
    read(ROOT/'predictions_complete.json');verify_predictions(d);config=read(d['configuration']['path'])
    def scoring(j,gpu):
        execute(['-m','radio_gs.query_field.public_lerf','score','--config',d['configuration']['path'],'--job',j['name'],
                 '--output',str(ROOT/'predictions')],ROOT/'logs/score'/(j['name']+'.log'),gpu)
        print('scored',j['name'],flush=True)
    lanes(config['jobs'],gpus,scoring)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation/evaluation.json') for j in config['jobs']]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','compile','predict','all','score']);p.add_argument('--name');p.add_argument('--gpus',default='1,2,3');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    else:
        d=design()
        if a.stage=='compile':
            j=next(j for j in d['jobs'] if j['name']==a.name)
            for r in j['receipts']:check(r)
            from radio_gs.query_field.compile_semantic_views import compile_scene
            compile_scene(j,ROOT/'fits'/j['name'])
        elif a.stage=='predict':
            read(ROOT/'construction_complete.json');predict_scene(next(j for j in d['jobs'] if j['name']==a.name),d)
        elif a.stage=='score':score(d,list(map(int,a.gpus.split(','))))
        else:run(d,list(map(int,a.gpus.split(','))))
