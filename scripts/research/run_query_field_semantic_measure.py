"""Fixed complete-menu semantic function study; compile before queries before labels."""
import argparse
import copy
import gc
from pathlib import Path
import subprocess

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from radio_gs.query_field.storage import load_field,save_field
from radio_gs.query_field.semantic_measure import SemanticMeasure
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes

ROOT=Path('output/query_field_20261009/semantic_measure_functions_v1')
PREVIOUS=Path('output/query_field_20261009/confidence_mixture_functions_v1')
PROTOCOL=Path('docs/experiments/2026-10-09-semantic-measure-protocol.md')
RULES=('mean','measure','reliable')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False)
    c=read(PREVIOUS/'confidence_mixture_function/cohort.json');v=read('configs/pipeline/system_v10.json')
    previous=read(PREVIOUS/'design.json');jobs=[]
    for old in c['scannet']:
        j=copy.deepcopy(old);name=j['name']
        j['observations']=next(x['observations'] for x in previous['fields'] if x['family']=='scannet' and x['scene']==name)
        source=read(Path(j['observations'])/'observations.json')
        legacy=next(x for x in v['jobs'] if x['benchmark']=='scannet' and x['name']==name)
        j['legacy_scene']=legacy['scene'];j['native_scene']=f'configs/consolidated/scannet_native/{name}.scene.json'
        spec=read(j['legacy_scene']);native=read(j['native_scene'])
        if native['source_frames']!=spec['source_frames']:raise ValueError('source authority differs')
        j['excluded_frames']=source['splits']['development_frames']
        j['full_frames']=[f for f in spec['source_frames'] if f not in j['excluded_frames']]
        if source['geometry']['sha256']!=spec['geometry']['sha256']:raise ValueError('geometry authority differs')
        j['receipts']=[receipt(p) for p in [j['legacy_scene'],j['native_scene'],Path(j['observations'])/'observations.json',
                         Path(j['field'])/'manifest.json',Path(j['field'])/'field.pt',spec['native_observation_field'],j['queries'],j['evaluation']]]
        jobs.append(j)
    if len(jobs)!=10:raise ValueError('full ten-scene cohort required')
    write(ROOT/'design.json',dict(jobs=jobs,encoder=c['encoder'],protocol=receipt(PROTOCOL),
        code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(Path(__file__)),
            receipt('radio_gs/querying/observation_reliability.py'),receipt('radio_gs/querying/native_observation_prior.py')],
        rules=RULES,primary='reliable_typed',training_steps=0,extra_member_labels_added=False,
        full_five_task=False,model_promotion=False,implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


def design():
    d=read(ROOT/'design.json')
    for r in [d['protocol']]+d['code']:check(r)
    return d


@torch.no_grad()
def compile_scene(j):
    root=ROOT/'fits'/j['name'];root.mkdir(parents=True,exist_ok=False)
    for r in j['receipts']:check(r)
    spec=read(j['legacy_scene']);native=read(j['native_scene'])
    source=torch.load(spec['native_observation_field'],map_location='cpu',weights_only=False)
    if not source.get('query_independent') or source['geometry']['sha256']!=spec['geometry']['sha256']:
        raise ValueError('compiled source contract differs')
    field,parent=load_field(j['field']);before={k:v.clone() for k,v in field.state_dict().items()}
    geo=torch.load(spec['geometry']['path'],map_location='cpu',weights_only=True)
    if not torch.equal(field.xyz,geo['xyz']):raise ValueError('carrier rows differ')
    offsets=[0];headers=[];keep=[]
    for frame in spec['source_frames']:
        p=next((Path(x)/f'frame_{frame:05d}.pt' for x in native['native_region_directories'] if (Path(x)/f'frame_{frame:05d}.pt').exists()),None)
        if p is None:raise ValueError('missing native header')
        row=torch.load(p,map_location='cpu',weights_only=False)
        if row.get('query_independent') is not True or int(row['frame'])!=frame:raise ValueError('native authority invalid')
        end=offsets[-1]+len(row['descriptors'])
        if not torch.equal(row['descriptors'].half(),source['descriptors'][offsets[-1]:end].half()):
            raise ValueError('descriptor concatenation changed')
        if frame in j['full_frames']:keep.extend(range(offsets[-1],end))
        offsets.append(end);headers.append(receipt(p))
    if offsets[-1]!=len(source['descriptors']):raise ValueError('unaccounted source atoms')
    a=sparse.csr_matrix((source['weights'].numpy(),source['columns'].numpy(),source['crow'].numpy()),shape=source['shape'])[:,keep].tocsr()
    a.eliminate_zeros();used=np.unique(a.indices);a=a[:,used].tocsr();a.sort_indices()
    rowmass=np.asarray(a.sum(1)).ravel();supported=rowmass>0
    a=(sparse.diags(1/np.maximum(rowmass,np.finfo(np.float32).tiny))@a).tocsr();a.sort_indices()
    descriptors=source['descriptors'][np.asarray(keep)[used]].clone()
    state=dict(descriptors=descriptors,pointer=torch.from_numpy(a.indptr.astype(np.int64)),
        index=torch.from_numpy(a.indices.astype(np.int64)),mass=torch.from_numpy(a.data.astype(np.float32)),supported=torch.from_numpy(supported))
    measure=SemanticMeasure(state,len(field.xyz)).cuda();replay=[]
    from radio_gs.querying.native_observation_prior import compiled_observation_prior
    reference=dict(descriptors=descriptors,crow=state['pointer'],columns=state['index'],weights=state['mass'],
                   shape=list(a.shape),supported=state['supported'])
    for shift in [0,1,2]:
        indices=(np.linspace(0,len(descriptors)-1,7).round().astype(int)+shift)%len(descriptors)
        q=descriptors[indices].cuda().float();normal=F.normalize(q,dim=1)
        fallback=torch.full((len(field.xyz),len(q)),1/len(q),device='cuda')
        for rule in ['measure','reliable']:
            refscene=dict(native_observation_field=spec['native_observation_field'])
            if rule=='reliable':refscene['native_observation_reliability']='entropy_information'
            logits,_=compiled_observation_prior(refscene,normal,state['supported'].cuda(),torch.logit(fallback),field=reference)
            expected=logits.sigmoid();actual=measure.probabilities(q,fallback,rule)
            error=float((expected-actual).abs().max());different=int((expected.argmax(1)!=actual.argmax(1)).sum())
            if error>2e-6:raise ValueError('semantic function replay mismatch: '+str(error))
            replay.append(dict(menu=shift,rule=rule,maximum_error=error,argmax_differences=different))
    field.semantic_measure=measure.cpu();field.semantic_measure_rule='reliable'
    assert all(torch.equal(field.state_dict()[k],v) for k,v in before.items())
    manifest=save_field(field,root/'field',dict(parent=receipt(Path(j['field'])/'field.pt'),
        legal_frames=j['full_frames'],excluded_frames=j['excluded_frames'],source=receipt(spec['native_observation_field']),
        source_headers=headers,query_free=True,labels_used=False,training_steps=0,uncompressed_semantic_measure=True,
        primary_rule='reliable',current_query_encoder='unchanged parent adapter; not v10 fp16-normalized wrapper',
        runtime_text_detection_transferred=False,local_auxiliary_channel_transferred=False))
    write(root/'construction.json',dict(scene=j['name'],frames=len(j['full_frames']),excluded=j['excluded_frames'],
        full_source_atoms=len(source['descriptors']),atoms=len(descriptors),nonzeros=a.nnz,
        supported=int(supported.sum()),parent_observed=int(field.observed.sum()),parent_unchanged=True,
        tensor_bytes=manifest['tensor_bytes'],parent_tensor_bytes=parent['tensor_bytes'],replay=replay,
        manifest=receipt(root/'field/manifest.json'),source_headers=headers))
    print('compiled',j['name'],len(descriptors),flush=True)


@torch.no_grad()
def predict_scene(j,d):
    from radio_gs.query_field.adapters import NativeQueryEncoder
    from radio_gs.query_field.query_program import query
    from radio_gs.query_field.category_evidence import corroborated_category_scores
    from radio_gs.querying.observation_readout import positive_support_rank
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False)
    queries=read(j['queries'])['queries'];encoder=NativeQueryEncoder(d['encoder'],'cuda')
    packets=[encoder.text(q['text'],task='union') for q in queries]
    del encoder;gc.collect();torch.cuda.empty_cache()
    field,manifest=load_field(ROOT/'fits'/j['name']/'field','cuda');native=[];member=[]
    for p in packets:
        r=query(field,p);native.append(r.semantic_scores);member.append(r.membership_scores)
    native=torch.stack(native,1);member=torch.stack(member,1);vectors=torch.stack([p.semantic.flatten() for p in packets])
    scores={'parent':corroborated_category_scores(native,member)}
    fallback=(100*native.float()).softmax(1)
    for rule in RULES:
        prior=field.semantic_measure.probabilities(vectors,fallback,rule)
        scores[rule+'_prior']=prior;scores[rule+'_typed']=positive_support_rank(prior,member)
    torch.save(dict(xyz=field.xyz.cpu(),observed=field.observed.cpu(),
        available=(field.observed|field.semantic_measure.supported).cpu(),queries=queries,
        vectors=vectors.cpu(),native=native.cpu(),member=member.cpu(),scores={k:v.cpu() for k,v in scores.items()}),out/'predictions.pt')
    write(out/'prediction.json',dict(scene=j['name'],queries=len(queries),field=manifest['files'],
        payload=receipt(out/'predictions.pt'),query=receipt(j['queries']),source_access=False,labels_read=False,
        all_complete_menu=True,primary=d['primary']))
    print('predicted',j['name'],flush=True)


def score(d):
    from scipy.spatial import cKDTree
    from plyfile import PlyData
    from radio_gs.scannet_constants import compute_split_metrics
    for r in read(ROOT/'predictions_complete.json')['receipts']:check(r)
    results=[]
    for j in d['jobs']:
        path=ROOT/'predictions'/j['name'];record=read(path/'prediction.json');check(record['payload'])
        data=torch.load(path/'predictions.pt',map_location='cpu',weights_only=True);spec=read(j['evaluation'])
        predictions={}
        for name,values in data['scores'].items():
            predictions[name]={}
            for split,ids in spec['splits'].items():
                columns=[next(i for i,q in enumerate(data['queries']) if q['class_id']==c) for c in ids]
                labels=np.asarray(ids)[values[:,columns].argmax(1).numpy()]
                labels[~data['observed' if name=='parent' else 'available'].numpy()]=0
                predictions[name][split]=labels
        vertices=PlyData.read(spec['ground_truth'])['vertex'].data
        xyz=np.stack([vertices[k] for k in ['x','y','z']],1);_,indices=cKDTree(data['xyz'].numpy()).query(xyz,workers=4)
        truth=np.asarray(vertices['label'])
        metrics={name:{split:compute_split_metrics(values[indices],truth,spec['splits'][split])
                       for split,values in splits.items()} for name,splits in predictions.items()}
        report=dict(scene=j['name'],metrics=metrics,predictions=record['payload'],ground_truth=receipt(spec['ground_truth']))
        write(path/'evaluation.json',report);results.append(report);print('scored',j['name'],flush=True)
    write(ROOT/'scoring_complete.json',dict(scenes=10,results=results))


def run(d,gpus):
    def build(j,gpu):execute(['-m','scripts.research.run_query_field_semantic_measure','compile','--name',j['name']],ROOT/'logs/build'/(j['name']+'.log'),gpu)
    lanes(d['jobs'],gpus,build)
    write(ROOT/'construction_complete.json',dict(receipts=[receipt(ROOT/'fits'/j['name']/'construction.json') for j in d['jobs']]))
    def predict(j,gpu):execute(['-m','scripts.research.run_query_field_semantic_measure','predict','--name',j['name']],ROOT/'logs/predict'/(j['name']+'.log'),gpu)
    lanes(d['jobs'],gpus,predict)
    write(ROOT/'predictions_complete.json',dict(receipts=[receipt(ROOT/'predictions'/j['name']/'prediction.json') for j in d['jobs']]))
    score(d)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','compile','predict','all','score']);p.add_argument('--name');p.add_argument('--gpus',default='1,2,3');args=p.parse_args()
    torch.set_num_threads(4)
    if args.stage=='prepare':prepare()
    else:
        d=design()
        if args.stage=='compile':compile_scene(next(j for j in d['jobs'] if j['name']==args.name))
        elif args.stage=='predict':
            read(ROOT/'construction_complete.json');predict_scene(next(j for j in d['jobs'] if j['name']==args.name),d)
        elif args.stage=='score':score(d)
        else:run(d,list(map(int,args.gpus.split(','))))
