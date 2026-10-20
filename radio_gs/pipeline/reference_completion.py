"""Signed reference repair -> two-frame propagation -> native-ray 3D response.

Consumes a freshly assembled source-only initializer, never historical answers
or evaluation annotations. Both N14 reference policies remain explicit; the
quality policy is the retained secondary recipe, not retroactively its primary.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image
import torch

from .reference_completion_contract import validate_profile, validate_geometry_training, sha
from radio_gs.querying.image_conditioned_response import lift_image_response
from radio_gs.querying.image_response_geometry import camera_args, render_responses, contributions, normalized_image_operator
from radio_gs.querying.reference_admission import reference_choices

ROOT = Path(__file__).resolve().parents[2]
WORKER_SOURCES = [
    'radio_gs/pipeline/reference_completion_worker.py', 'radio_gs/querying/reference_transport.py',
    'radio_gs/querying/reference_admission.py', 'radio_gs/querying/reference_completion.py',
    'radio_gs/querying/signed_reference_affinity.py', 'radio_gs/querying/image_conditioned_response.py',
    'radio_gs/scripts/build_sam3_foundation_cache.py',
    '/root/external/sam3/sam3/model_builder.py', '/root/external/sam3/sam3/model/sam3_image.py',
    '/root/external/sam3/sam3/model/sam3_image_processor.py',
    '/root/external/sam3/sam3/model/sam1_task_predictor.py',
    '/root/external/sam3/sam3/model/sam3_tracking_predictor.py',
    '/root/external/sam3/sam3/model/sam3_tracker_base.py',
    '/root/external/sam3/sam3/model/utils/sam2_utils.py',
    '/root/external/sam3/sam3/model/utils/sam1_utils.py',
]


def read(path):return json.loads(Path(path).read_text())


def worker_sources(profile):
    dino=Path(profile['dino_repository'])/'dinov2'
    return WORKER_SOURCES+[str(p) for folder in ['models','layers','hub'] for p in sorted((dino/folder).glob('*.py'))]


def completed_readout(initial, operator, image, valid, profile, worker):
    """Same complete-response solve; independent selected set remains explicitly inherited."""
    if worker['choices'] != reference_choices(worker['decoder']):
        raise ValueError('reference choice differs from declared source-only policy')
    if (worker['target_gt_available'] or worker['reference_gt_available'] or
            not worker['negative_scribble_used'] or not worker['sam_eval'] or not worker['dino_eval']):
        raise ValueError('decoder input/evaluation contract differs')
    old, observed = initial['scores'].float(), initial['semantic_observed'].bool()
    if old.shape != (len(observed), 1) or torch.count_nonzero(old[~observed]):
        raise ValueError('initializer response/support domain differs')
    y = torch.as_tensor(image, dtype=torch.float32).reshape(-1, 1)
    if valid.dtype != torch.bool or valid.shape != y.shape:
        raise ValueError('image visibility domain differs')
    confidence = (2*y-1).abs() * valid
    value, lifting = lift_image_response(old, operator, y, confidence, profile['solver_steps'])
    from radio_gs import evidence_core
    if evidence_core.enabled():
        from radio_gs.evidence_core.core import _record
        _record('reference_completion_native_writeback')
    if lifting['unsupported_changed'] or torch.count_nonzero(value[~observed]):
        raise ValueError('native writeback modified unsupported elements')
    result = dict(initial, scores=value,
        reference_completion=dict(profile='N14', policy=profile['policy'],
            historical_primary_policy='feasible', selected_reference=worker['choices'][profile['policy']],
            permissions=profile['permissions'], target_frame=profile['target_frame'],
            native_wh=profile['native_wh'], threshold=profile['threshold'],
            view_conditioned=True, initializer='fresh_source_assisted_response',
            selection_profile='inherited_source_only_selected_set_unchanged'),
        view_conditioned_image_response=torch.as_tensor(image, dtype=torch.float32))
    if 'group_readouts' in result:
        result['initializer_group_readouts']=result.pop('group_readouts')
    return result, lifting


@torch.inference_mode()
def run(scene_path, query_path, profile_path, initial_path, output, final_output, sam_python):
    from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
    from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
    started=time.monotonic();scene,query,profile=map(read,[scene_path,query_path,profile_path])
    validate_profile(profile,scene,query,verify_assets=True)
    validate_geometry_training(profile,read(profile['geometry_fit']['path']))
    initial=torch.load(initial_path,map_location='cpu',weights_only=False)
    if Path(initial['generation_plan']).resolve() != Path(initial_path).resolve().parent/'execution.json':
        raise ValueError('initializer must belong to this fresh pipeline execution')
    binding=initial['semantic_geometry']
    if (binding['sha256']!=scene['geometry']['sha256'] or
            Path(binding['path']).resolve()!=Path(scene['geometry']['path']).resolve()):
        raise ValueError('initializer uses a different geometry')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(profile['gpu_memory_fraction'])
    geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
    gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256'])
    old,observed=initial['scores'].float(),initial['semantic_observed'].bool()
    args=camera_args(geo,gp,profile['target_frame'],profile['native_wh'])
    prior=render_responses(args,[old],observed)[...,0]
    if shutil.disk_usage('/dev/shm').free < 96*2**20:
        raise RuntimeError('insufficient private worker transient capacity')
    sources=worker_sources(profile)
    worker_hashes={p:sha(ROOT/p) for p in sources}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='radiogs-reference-completion-',dir='/dev/shm') as temporary:
        root=Path(temporary);np.save(root/'prior.npy',prior.numpy(),allow_pickle=False)
        assets={k:v['path'] for k,v in profile['inputs'].items()}
        Image.open(assets['target_rgb']).convert('RGB').resize(tuple(profile['native_wh']),Image.Resampling.BILINEAR).save(root/'rgb.png')
        request=dict(**assets,checkpoint=profile['image_checkpoint']['path'],
            dino_checkpoint=profile['dino_checkpoint']['path'],dino_repository=profile['dino_repository'],
            points_per_sign=profile['points_per_sign'],gpu_memory_fraction=profile['gpu_memory_fraction'],
            source_paths=sources)
        (root/'request.json').write_text(json.dumps(request,indent=2)+'\n')
        gc.collect();torch.cuda.empty_cache()
        subprocess.run([sam_python,'-m','radio_gs.pipeline.reference_completion_worker','--request',str(root)],
            cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),check=True)
        worker=read(root/'result.json')
        if worker['source_hashes']!=worker_hashes or any(sha(ROOT/p)!=h for p,h in worker_hashes.items()):
            raise ValueError('worker source changed during inference')
        image=np.load(root/(profile['policy']+'_image.npy'),allow_pickle=False)
        if image.shape!=tuple(profile['native_wh'][::-1]):raise ValueError('decoder image grid differs')
        transient_bytes=sum(p.stat().st_size for p in root.iterdir())
        if transient_bytes>64*2**20:raise RuntimeError('reference worker transient budget exceeded')
    hit,alpha_error=contributions(geo,args,profile['footprint_backend'])
    operator,valid=normalized_image_operator(hit,observed,prior.numel())
    check=torch.sparse.mm(operator,old);check[~valid]=0
    operator_error=float((check.reshape(prior.shape)-prior).abs().max())
    if operator_error>=1e-4:raise ValueError('native operator differs from complete-scene rendering')
    del hit;torch.cuda.empty_cache()
    result,lifting=completed_readout(initial,operator,image,valid,profile,worker)
    rendered=render_responses(args,[result['scores']],observed)[...,0]
    if not bool(torch.isfinite(rendered).all()):raise ValueError('nonfinite native response')
    result['reference_completion']['profile_sha256']=sha(profile_path)
    torch.save(result,final_output)
    report=dict(schema='radiogs.reference_completion_run.v1',profile=profile,decoder=worker,lifting=lifting,
        input_permissions=profile['permissions'],initial_sha256=sha(initial_path),final_sha256=sha(final_output),
        historical_predictions_read=False,evaluation_annotations_read=False,geometry_updated=False,
        native_operator_max_error=operator_error,alpha_max_error=alpha_error,
        transient_worker_bytes=transient_bytes,worker_transients_released=True,
        seconds=time.monotonic()-started,source_sha256={p:sha(ROOT/p) for p in sources})
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','query','profile','initial','output','final-output','sam-python'):
        parser.add_argument('--'+key,required=True)
    a=parser.parse_args()
    run(a.scene,a.query,a.profile,a.initial,a.output,a.final_output,a.sam_python)


if __name__=='__main__':main()
