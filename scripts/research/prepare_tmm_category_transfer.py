"""Replay original category providers before model/benchmark label access.

All ten complete candidate pools and their signatures are sealed together.
Dense arrays live in a bounded private tmpfs bank for subsequent fixed-model
evaluation, never in the persistent study directory.
"""
import argparse
import gc
from pathlib import Path
import time
import numpy as np
import torch
from radio_gs.querying.response_study_adapters import read,load,readout_path
from radio_gs.querying.category_response_transfer import complete_menu_candidates,menu_signature
from radio_gs.query_field.bounded_experiment import BoundedExperiment,tree_bytes
from scripts.research.replay_tmm_response import STUDY,POLICY,sha
from scripts.research.tmm_ram_predictions import RamPredictionBank

PROVIDERS=('dual_native_source','dual_native','source_readout')


@torch.inference_mode()
def prepare(job,run,memory):
    start=time.monotonic(); scene=read(job['scene']); auxiliary_scene=read(scene['local_auxiliary_scene'])
    auxiliary_path=Path(scene['local_auxiliary_scene']).parent/'query/typed_readout.pt'
    original_scene_path=Path('configs/consolidated/scannet_native')/(job['name']+'.scene.json')
    original_scene=read(original_scene_path)
    paths=[Path(job['scene']),Path(scene['local_auxiliary_scene']),readout_path(job),auxiliary_path,
        original_scene_path,Path(scene['native_observation_field']),Path(auxiliary_scene['native_observation_field'])]
    field=load(scene['native_observation_field']); auxiliary_field=load(auxiliary_scene['native_observation_field'])
    offsets=[0]
    if original_scene['source_frames'] != scene['source_frames']:
        raise ValueError('native and source view identities changed')
    for frame in original_scene['source_frames']:
        path=Path(original_scene['native_region_directories'][0])/f'frame_{frame:05d}.pt'
        native=load(path); descriptor=native['descriptors']; end=offsets[-1]+len(descriptor)
        if native['frame'] != frame or not torch.equal(descriptor.half(),field['descriptors'][offsets[-1]:end]):
            raise ValueError('retained native region/view ordering differs')
        offsets.append(end); paths.append(path)
    data=load(readout_path(job)); auxiliary=load(auxiliary_path)
    print('category replay',job['name'],'views',len(offsets)-1,flush=True)
    candidates,scores,compatibility=complete_menu_candidates(data,scene,auxiliary_scene,auxiliary,field,auxiliary_field)
    x,xp,evidence=menu_signature(candidates,data['semantic_observed'].numpy(),field,offsets,scores)
    paths += [Path(v[k]) for v in scene['source_evidence']['views'] for k in ('identity','contributions')]
    report=dict(job=job,columns=data['texts'],class_ids=[q['class_id'] for q in data['queries']],
        providers=PROVIDERS,features=x.tolist(),permuted_features=xp.tolist(),evidence=evidence,
        compatibility=compatibility,views=len(offsets)-1,regions=offsets[-1],ground_truth_read=False,
        model_read=False,source_images_read=False,source_evidence_reused=True,
        original_gaussian_carrier=True,all_applicable_teachers_retained=True,
        independent_mesh_signature_domain_equivalence=False,input_hashes={str(p):sha(p) for p in paths},
        prediction_arrays={name:memory.add(run,job['name']+'_'+name,torch.from_numpy(value))
                           for name,value in zip(PROVIDERS,candidates)},seconds=time.monotonic()-start)
    run.json(job['name']+'.json',report)
    print('category pool ready',job['name'],'seconds',round(report['seconds'],1),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='category_transfer_inputs_v2')
    p.add_argument('--resume',action='store_true');p.add_argument('--scenes',nargs='+')
    args=p.parse_args();torch.set_num_threads(4)
    jobs=[j for j in read('configs/pipeline/system_v10.json')['jobs'] if j['benchmark']=='scannet']
    if args.scenes and not set(args.scenes)<={j['name'] for j in jobs}:
        raise ValueError('undeclared category scene')
    sources=[__file__,'radio_gs/querying/category_response_transfer.py','radio_gs/querying/native_observation_prior.py',
        'radio_gs/querying/local_element_fusion.py','radio_gs/querying/typed_evidence.py',
        'radio_gs/querying/observation_reliability.py','radio_gs/querying/response_replacement.py',
        'scripts/research/tmm_ram_predictions.py','scripts/research/run_tmm_support_agreement.py']
    hashes={s:sha(s) for s in sources}
    run=BoundedExperiment(STUDY,args.run,POLICY,resume=args.resume);run.preflight(2*2**20)
    if args.resume:
        manifest=read(run.root/'manifest.json')
        if manifest['source_hashes']!=hashes or manifest['jobs']!=jobs:
            raise ValueError('sealed preparation source/protocol changed')
        memory=RamPredictionBank(manifest['ram_directory'])
    else:
        memory=RamPredictionBank()
        run.json('manifest.json',dict(jobs=jobs,source_hashes=hashes,providers=PROVIDERS,ram_directory=str(memory.root),
            ground_truth_read=False,model_read=False,benchmark_parameters_fitted=False,default_changed=False,
            visibility_domain='union of retained native region support, not full geometric visibility',
            independent_mesh_signature_domain_equivalence=False))
        run.json('source_snapshot.json',{s:dict(text=Path(s).read_text(),sha256=hashes[s]) for s in sources})
    for job in jobs:
        if args.scenes and job['name'] not in args.scenes:continue
        path=run.root/(job['name']+'.json')
        if path.exists():
            report=read(path)
            for name in PROVIDERS:memory.read(run,job['name']+'_'+name,len(report['columns']))
            continue
        prepare(job,run,memory);gc.collect();torch.cuda.empty_cache()
    reports=[read(run.root/(j['name']+'.json')) for j in jobs if (run.root/(j['name']+'.json')).exists()]
    if len(reports)==10 and not (run.root/'summary.json').exists():
        run.json('summary.json',dict(complete=True,jobs=10,ground_truth_read=False,model_read=False,
            report_hashes={r['job']['name']+'.json':sha(run.root/(r['job']['name']+'.json')) for r in reports},
            original_incumbents_preserved=True,transient_prediction_bytes=memory.bytes,durable_dense_prediction_bytes=0,
            prediction_bank_retained_for_fixed_model_evaluation=True,study_bytes=tree_bytes(STUDY)))
    print('category transfer inputs complete',len(reports),'/ 10',flush=True)


if __name__=='__main__':main()
