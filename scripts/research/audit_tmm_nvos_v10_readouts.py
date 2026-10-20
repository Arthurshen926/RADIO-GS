"""Audit both retained in-house NVOS outputs with their existing readout rules."""
import gc
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from gsplat import rasterization

from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from scripts.research.audit_tmm_protocol_correctness import REPO, SYSTEM, read, load, binary_counts
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.tmm_ram_predictions import RamPredictionBank

CONFIG='configs/research/nvos_readout_inheritance.json'


@torch.inference_mode()
def predict(job,run,bank):
    name=job['name'];path=SYSTEM/'nvos'/name/'query/typed_readout.pt'
    d=load(path);scene=read(job['scene']);spec=read(job['evaluation'])
    assert spec['target_frame'] not in scene['source_frames']
    gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256'])
    cam=gp.cameras[spec['target_frame']];width,height=spec['raster_wh']
    k=cam.intrinsic.cuda().clone();k[0]*=width/cam.width;k[1]*=height/cam.height
    inputs=[path,job['scene'],job['evaluation'],scene['camera_report']];seals={}
    for arm,prefix,key in [('semantic_response','semantic','scores'),
                           ('independent_set_projection','selection','selection_scores')]:
        binding=d[prefix+'_geometry'];inputs.append(binding['path'])
        geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
        values=d[key].cuda();observed=d[prefix+'_observed'].cuda()
        opacity=geo.opacities.flatten()
        if arm=='independent_set_projection':opacity=opacity*(values[:,0]>.5)*observed
        image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,
            opacities=opacity,colors=torch.cat([values,observed.float()[:,None]],1),
            viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=width,height=height,
            near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False)
        if arm=='independent_set_projection':response=alpha[0,...,0].cpu()
        else:
            mass=image[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
            response=torch.where(valid,image[0,...,0]/mass.clamp_min(1e-8),0).cpu()
        seals[name+'_'+arm]=bank.add(run,name+'_'+arm,response.reshape(-1,1))
        del geo,values,observed,image,alpha,response
        gc.collect();torch.cuda.empty_cache()
    report=dict(job=job,predictions=seals,input_hashes={str(p):sha(p) for p in inputs},
                target_gt_read=False,target_rgb_used=False)
    run.json(name+'_prediction.json',report);print('sealed readouts',name,flush=True)
    return report


def main():
    os.chdir(REPO);torch.set_num_threads(2);config=read(CONFIG)
    torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    run=BoundedExperiment(STUDY,config['run'],POLICY);run.preflight(config['maximum_new_durable_bytes'])
    bank=RamPredictionBank(limit=config['maximum_volatile_bytes'])
    jobs=read('configs/pipeline/system_v10.json')['jobs'];nvos=[j for j in jobs if j['benchmark']=='nvos']
    sources=[__file__,CONFIG,'scripts/research/evaluate_reference_instance.py']
    run.json('manifest.json',dict(config=config,jobs=nvos,source_hashes={p:sha(p) for p in sources},
        torch_version=torch.__version__,volatile_prediction_directory=str(bank.root)))
    run.json('source_snapshot.json',{p:dict(sha256=sha(p),text=Path(p).read_text()) for p in sources})
    try:
        reports=[predict(j,run,bank) for j in nvos]
        run.json('predictions_complete.json',dict(jobs=8,target_gt_read=False,
            report_hashes={r['job']['name']:sha(run.root/(r['job']['name']+'_prediction.json')) for r in reports},
            predictions={k:v for r in reports for k,v in r['predictions'].items()}))
        barrier=STUDY/config['scoring_barrier'];started=time.monotonic()
        print('all v10 readouts sealed; waiting for joint NVOS prediction barrier',flush=True)
        while not barrier.exists():
            if (barrier.parent/'failure.json').exists() or time.monotonic()-started>7200:
                raise RuntimeError('coverage prediction barrier failed or timed out; no labels opened')
            time.sleep(5)
        rows={};gt_hashes={}
        for job in nvos:
            name=job['name'];spec=read(job['evaluation']);gt=np.asarray(Image.open(spec['gt_mask']))
            gt_hashes[spec['gt_mask']]=sha(spec['gt_mask']);rows[name]={}
            truths={'retained_gt_gt127':gt>127,'ludvig_gt_maximum':(gt//gt.max()).astype(bool)}
            for arm in config['arms']:
                response=bank.read(run,name+'_'+arm,1).reshape(spec['raster_wh'][1],spec['raster_wh'][0]).numpy()
                response=cv2.resize(response,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR)
                binary=(np.floor(np.clip(response,0,1)*255+.5)>10) if arm=='independent_set_projection' else response>=.5
                rows[name][arm]={track:dict(**binary_counts(binary,truth),
                    average_precision=binary_ranking_metrics(torch.from_numpy(response).flatten(),
                        torch.from_numpy(truth).flatten())['average_precision']) for track,truth in truths.items()}
            old=read(SYSTEM/'nvos'/name/'result.json')['metrics']
            scored=rows[name]['semantic_response']['retained_gt_gt127']
            assert abs(scored['iou']-old['iou'])<1e-5 and abs(scored['average_precision']-old['ap'])<1e-5
            run.json(name+'_evaluation.json',rows[name])
        macro={t:{a:{m:float(np.mean([r[a][t][m] for r in rows.values()]))
                      for m in ('iou','average_precision','precision','recall')} for a in config['arms']}
               for t in truths}
        run.json('summary.json',dict(complete=True,rows=rows,macro=macro,gt_hashes=gt_hashes,
            prediction_seal_sha256=sha(run.root/'predictions_complete.json'),
            joint_barrier_sha256=sha(barrier),new_method_gain=False,default_changed=False,
            inherited_output_audit=True,per_task_winners=False,independent_test=False,
            output_metric_change_explicit=True,durable_dense_prediction_bytes=0))
        print(json.dumps(macro,indent=2),flush=True)
    except Exception as error:
        run.json('failure.json',dict(type=type(error).__name__,message=str(error),default_changed=False))
        raise
    finally:bank.close()


if __name__=='__main__':main()
