"""Prepare both whole-cohort N14 input profiles; no teacher or prediction calls.

Dataset/evaluation metadata supplies target frame and RGB identity only. GT mask
pixels and historical predictions are never opened. Other 23 jobs are preserved.
"""
import argparse
import copy
import json
from pathlib import Path

from radio_gs.pipeline.reference_completion_contract import PERMISSIONS, validate_profile, validate_geometry_training, sha
from radio_gs.pipeline.runtime import PipelineConfig, build_plan

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',default='output/shared_evidence_20261018/region_provenance_20261019_v1/recipe.json')
    parser.add_argument('--dataset',default='/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    recipe=read(a.recipe);dataset=read(a.dataset)
    settings=read('configs/research/nvos_reference_admission.json');teachers=read('configs/research/nvos_signed_affinity.json')
    hashes={}
    def asset(path):
        path=str(Path(path).resolve())
        if path not in hashes:hashes[path]=sha(path)
        return dict(path=path,sha256=hashes[path])
    model_assets={k:asset(teachers[k]) for k in ['image_checkpoint','dino_checkpoint']}
    paths=[];plans=[]
    for policy in ['quality','feasible']:
        folder=a.output/policy;folder.mkdir();derived=copy.deepcopy(recipe)
        for job in derived['jobs']:
            if job['benchmark']!='nvos':continue
            scene,query,evaluation=map(read,[job['scene'],job['queries'],job['evaluation']])
            ds=next(s for s in dataset['scenes'] if s['scene_id']==job['name'])
            target=next(f for f in ds['frames'] if int(f['canonical_index'])==evaluation['target_frame'])
            if target['frame_id'] not in ds['evaluation_frame_ids']:raise ValueError('target frame split differs')
            inputs={k:asset(query[k]) for k in ['reference_rgb','reference_mask','positive_scribble']}
            inputs['negative_scribble']=asset(ds['prompt']['negative_path']);inputs['target_rgb']=asset(target['rgb_path'])
            for sign in ['positive','negative']:
                if inputs[sign+'_scribble']['sha256']!=dataset['protocol']['prompt_asset_sha256'][job['name']][sign]:
                    raise ValueError('official scribble asset differs')
            profile={k:settings[k] for k in ['native_wh','solver_steps','threshold','footprint_backend','points_per_sign','gpu_memory_fraction']}
            profile.update(schema='radiogs.signed_reference_completion.v1',policy=policy,
                target_frame=evaluation['target_frame'],inputs=inputs,permissions=PERMISSIONS,
                dino_repository=teachers['dino_repository'],**model_assets,
                geometry_fit=asset(Path(scene['geometry']['path']).parent/'fit.json'))
            validate_profile(profile,scene,query)
            validate_geometry_training(profile,read(profile['geometry_fit']['path']))
            path=folder/(job['name']+'.json');path.write_text(json.dumps(profile,indent=2)+'\n');paths.append(path)
            job['reference_completion']=str(path.resolve())
            job['prompt_contract']='N14: original signed scribbles, inherited positive seed, explicit observed target RGB'
            job['input_track']='frozen source-only geometry; query-time target RGB allowed; no target GT'
            plan=build_plan(PipelineConfig(scene=job['scene'],queries=job['queries'],
                output=str(ROOT/'tmp/not-executed-n14'/policy/job['name']),reference_tracks=job['reference_tracks'],
                reference_completion=str(path)))
            plans.append(dict(name=job['name'],policy=policy,commands=plan['commands'],
                target_rgb_allowed=plan['target_rgb_allowed'],commands_executed=False))
        recipe_path=a.output/(policy+'.recipe.json');recipe_path.write_text(json.dumps(derived,indent=2)+'\n');paths.append(recipe_path)
    report=dict(schema='radiogs.reference_completion_profiles.v1',policies=['quality','feasible'],
        historical_primary='feasible',retained_secondary='quality',jobs_per_recipe=len(recipe['jobs']),
        nvos_tasks=sum(j['benchmark']=='nvos' for j in recipe['jobs']),input_sha256=hashes,
        profiles_sha256={str(p):sha(p) for p in paths},plans=plans,model_calls=0,
        evaluation_metadata_read=True,ground_truth_pixels_read=False,historical_predictions_read=False,
        gpu_end_to_end_validated=False,default_changed=False)
    (a.output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(profiles=len(plans),jobs_per_recipe=len(recipe['jobs']),assets=len(hashes),model_calls=0)))


if __name__=='__main__':main()
