"""CPU input, source-decision and route audit; no GPU or benchmark predictions."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image

from radio_gs.pipeline.reference_completion_contract import validate_profile, validate_geometry_training, sha
from radio_gs.pipeline.reference_completion import worker_sources
from radio_gs.pipeline.runtime import PipelineConfig, build_plan
from radio_gs.querying.reference_completion import signed_scribble_points
from radio_gs.querying.reference_admission import reference_choices
from scripts.research.run_shared_evidence_migration import validate_extension_recipe

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())


def binary(path):
    x=np.asarray(Image.open(path))
    return np.any(x>127,axis=2) if x.ndim==3 else x>127


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles',default='configs/pipeline/reference_completion_20261019')
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    preparation=read(Path(a.profiles)/'preparation.json')
    base=read('output/shared_evidence_20261018/region_provenance_20261019_v1/recipe.json')
    paths=[Path(__file__),Path('scripts/research/run_shared_evidence_migration.py'),
           Path('scripts/research/evaluate_reference_instance.py'),Path('radio_gs/evaluation/reference_response.py'),
           Path('tests/test_reference_completion_pipeline.py')]
    for folder in ['radio_gs/pipeline','radio_gs/querying','radio_gs/evidence_core']:
        paths.extend((ROOT/folder).glob('*.py'))
    source_hashes={str(p):sha(p) for p in paths};rows=[];input_hashes={}
    # Recheck actual declared image/model/geometry-fit contents once, shared by both policies.
    for p,h in preparation['input_sha256'].items():
        if sha(p)!=h:raise ValueError('prepared input asset changed')
        input_hashes[p]=h
    for p,h in preparation['profiles_sha256'].items():
        if sha(p)!=h:raise ValueError('prepared profile/recipe changed')
    for policy in ['quality','feasible']:
        recipe=read(Path(a.profiles)/(policy+'.recipe.json'));validate_extension_recipe(recipe,base)
        for job in recipe['jobs']:
            if job['benchmark']!='nvos':continue
            scene,query,profile=map(read,[job['scene'],job['queries'],job['reference_completion']])
            permissions=validate_profile(profile,scene,query)
            validate_geometry_training(profile,read(profile['geometry_fit']['path']))
            masks={k:binary(profile['inputs'][k]['path']) for k in ['positive_scribble','negative_scribble','reference_mask']}
            if any(m.shape!=tuple(profile['native_wh'][::-1]) for m in masks.values()):raise ValueError('native reference grids differ')
            points=signed_scribble_points(masks['positive_scribble'],masks['negative_scribble'],profile['points_per_sign'])
            old_path=Path('output/tmm_response_20261018/study/nvos_reference_admission_v1')/(job['name']+'_prediction.json')
            decoder=read(old_path)['decoder'];input_hashes[str(old_path)]=sha(old_path)
            decisions=reference_choices(decoder['decoder'])
            if decisions!=decoder['choices']:raise ValueError('source selection differs from recorded N14')
            if not (np.array_equal(points['point_coords'],decoder['point_coords']) and np.array_equal(points['point_labels'],decoder['point_labels'])):
                raise ValueError('signed points differ from recorded N14')
            for key,sign in [('original_positive_pixels','positive_scribble'),('original_negative_pixels','negative_scribble')]:
                if decoder['decoder']['semantic_memory'][key]!=int(masks[sign].sum()):raise ValueError('signed input counts differ')
            plan=build_plan(PipelineConfig(scene=job['scene'],queries=job['queries'],
                output=str(ROOT/'tmp/not-executed-n14-audit'/policy/job['name']),
                reference_tracks=job['reference_tracks'],reference_completion=job['reference_completion']))
            if not plan['target_rgb_allowed'] or plan['commands'][-1][2]!='radio_gs.pipeline.reference_completion':
                raise ValueError('complete provider is not consumed by pipeline')
            for path in worker_sources(profile):source_hashes[path]=sha(ROOT/path)
            rows.append(dict(name=job['name'],policy=policy,selected_reference=decisions[policy],
                signed_points_exact=True,source_decision_exact=True,permissions=permissions,
                source_positive_pixels=int(masks['positive_scribble'].sum()),source_negative_pixels=int(masks['negative_scribble'].sum()),
                fresh_initializer_planned=True,complete_provider_planned=True,model_calls=0))
    def body(path,name):
        return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
    worker_equal=body('scripts/research/decode_tmm_nvos_reference_admission.py','main')==body('radio_gs/pipeline/reference_completion_worker.py','main')
    if not worker_equal:raise ValueError('canonical worker arithmetic differs from historical body')
    unchanged=all(sha(ROOT/p)==h for p,h in source_hashes.items())
    if not unchanged:raise ValueError('source changed during CPU audit')
    result=dict(schema='radiogs.reference_completion_cpu_audit.v1',passed=True,rows=rows,
        source_sha256=source_hashes,profiles_sha256=preparation['profiles_sha256'],input_sha256=input_hashes,
        sources_unchanged_during_audit=True,worker_function_ast_equal=worker_equal,
        full_benchmark_evaluation=False,gpu_end_to_end_validated=False,
        historical_decoder_metadata_read=True,historical_prediction_arrays_read=False,ground_truth_pixels_read=False,
        limits='Input/source-policy/route checks and unchanged worker body; no fresh DINO/SAM, rendering, or N14 accuracy measured.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(profiles=len(rows),passed=True,worker_function_ast_equal=worker_equal,model_calls=0)))


if __name__=='__main__':main()
