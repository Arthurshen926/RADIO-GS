"""All-task N16 score audit. No new candidate, thresholds or persistent arrays."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import cv2
import numpy as np
from PIL import Image
from scipy.special import expit
from sklearn.metrics import average_precision_score
import torch

from radio_gs.evaluation.response_score_precision import stable_probability_readout, saturation_counts
from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from scripts.research.audit_tmm_protocol_correctness import REPO, SYSTEM, read, load, binary_counts
from scripts.research.replay_tmm_response import STUDY, sha
from scripts.research.run_tmm_nvos_target_redecode import WORKER_SOURCES, write
from scripts.research.run_tmm_nvos_image_response import MANIFEST
from scripts.research.run_tmm_support_agreement import array_sha
from scripts.research.tmm_ram_predictions import RamPredictionBank

CONFIG = 'configs/research/nvos_score_precision.json'
OBSERVER = 'scripts/research/decode_tmm_nvos_score_precision.py'
EXTERNAL = ['/root/external/sam3/sam3/model/sam1_task_predictor.py',
            '/root/external/sam3/sam3/model/sam3_image.py',
            '/root/external/sam3/sam3/model/sam3_tracking_predictor.py']


def matching_raw(root, captures, kind, expected):
    matches = []
    for r in captures:
        if r['kind'] != kind or r['shape'] != list(expected.shape): continue
        z = np.load(root/r['file'], allow_pickle=False)
        if np.array_equal(torch.from_numpy(z).sigmoid().numpy(), expected): matches.append(z)
    assert matches, 'no exact raw-logit match'
    assert all(np.array_equal(matches[0], z) for z in matches), 'ambiguous saturated-logit match'
    return matches[0]


def replay(job, run, bank, config, parent, pm, dataset, access):
    started = time.monotonic(); name = job['name']; previous = read(parent/(name+'_prediction.json'))
    query, spec = read(job['queries']), read(job['evaluation'])
    ds = next(s for s in dataset['scenes'] if s['scene_id'] == name)
    frame = next(f for f in ds['frames'] if int(f['canonical_index']) == spec['target_frame'])
    ancestor = read(STUDY/pm['config']['parent_run']/'manifest.json')['config']
    access['rgb'] = True
    base = read(STUDY/pm['config']['parent_run']/(name+'_prediction.json'))
    for path, h in base['input_hashes'].items(): assert sha(path) == h
    access['rgb'] = False
    prior = load(SYSTEM/'nvos'/name/'evaluation/response.pt')['response'].numpy()
    assert array_sha(prior.reshape(-1,1)) == previous['predictions']['v10']['sha256']
    with tempfile.TemporaryDirectory(prefix='radiogs_score_audit_', dir='/dev/shm') as tmp:
        root = Path(tmp); np.save(root/'prior.npy', prior, allow_pickle=False)
        access['rgb'] = True
        Image.open(frame['rgb_path']).convert('RGB').resize((1008,756), Image.Resampling.BILINEAR).save(root/'rgb.png')
        access['rgb'] = False
        request = dict(checkpoint=ancestor['image_checkpoint'], gpu_memory_fraction=config['gpu_memory_fraction'],
            source_paths=WORKER_SOURCES, reference_mask=query['reference_mask'], reference_rgb=query['reference_rgb'],
            positive_scribble=query['positive_scribble'], negative_scribble=ds['prompt']['negative_path'],
            dino_repository=ancestor['dino_repository'], dino_checkpoint=ancestor['dino_checkpoint'],
            target_rgb=frame['rgb_path'], points_per_sign=pm['config']['points_per_sign'],
            context_fraction=pm['config']['context_fraction'])
        (root/'request.json').write_text(json.dumps(request))
        subprocess.run(['/root/miniconda3/envs/sam3/bin/python', OBSERVER, '--request', str(root)], check=True)
        worker, captures = read(root/'result.json'), read(root/'raw_records.json')
        assert worker['decoder'] == previous['decoder']['decoder']
        assert worker['choices'] == previous['decoder']['choices']
        assert worker['refinement'] == previous['decoder']['refinement']
        assert worker['source_hashes'] == pm['worker_source_hashes']
        probabilities = {a: np.load(root/(a+'.npy'),allow_pickle=False) for a in config['arms']}
        raw = {'quality_image': matching_raw(root,captures,'memory',probabilities['quality_image'])}
        reconstructed = {'quality_image': torch.from_numpy(raw['quality_image']).sigmoid().numpy()}
        for arm in ['full_image','crop_image']:
            box = worker['refinement'][arm.split('_')[0]]['box']
            raw[arm] = raw['quality_image'].copy()
            reconstructed[arm] = reconstructed['quality_image'].copy()
            if box is not None:
                x0,y0,x1,y1 = box
                local_raw = matching_raw(root,captures,'image',probabilities[arm][y0:y1,x0:x1])
                raw[arm][y0:y1,x0:x1] = local_raw
                reconstructed[arm][y0:y1,x0:x1] = torch.from_numpy(local_raw).sigmoid().numpy()
        entries = {}
        for arm in config['arms']:
            p,z = probabilities[arm],raw[arm]
            assert array_sha(p.reshape(-1,1)) == previous['predictions'][arm]['sha256'], (name,arm,'N16 replay drift')
            # Preserve the worker's local sigmoid THEN paste operation order.
            # A whole-canvas sigmoid can use different vector/tail rounding.
            assert np.array_equal(reconstructed[arm],p)
            whole_canvas = torch.from_numpy(z).sigmoid().numpy()
            numerical_delta = float(np.max(np.abs(whole_canvas-p)))
            assert numerical_delta <= 2e-7 and np.array_equal(z>=0,p>=.5)
            entries[arm] = dict(probability=bank.add(run,name+'_'+arm,torch.from_numpy(p.reshape(-1,1))),
                logits=bank.add(run,name+'_'+arm+'_logits',torch.from_numpy(z.reshape(-1,1))),
                reconstruction_exact=True,whole_canvas_sigmoid_max_delta=numerical_delta,
                min_logit=float(z.min()),max_logit=float(z.max()),one_pixels=int((p==1).sum()),zero_pixels=int((p==0).sum()),
                unique_probabilities=int(np.unique(p).size), unique_logits=int(np.unique(z).size))
        scratch_bytes = tree_bytes(root); assert scratch_bytes < 128*2**20
    report = dict(name=name,entries=entries,decoder_exact=True,all_probability_hashes_exact=True,
        parent_report_sha256=sha(parent/(name+'_prediction.json')),target_gt_read=False,
        scratch_directory=str(root),scratch_bytes=scratch_bytes,seconds=time.monotonic()-started)
    write(run,name+'_prediction.json',report)
    print('sealed',name,'3 exact probabilities + 3 raw logits',round(report['seconds'],1),'s',flush=True)
    return report


def metrics(score, truth, threshold):
    b = score >= threshold
    result = binary_counts(b,truth)
    tn,fp,fn,tp = np.bincount((2*truth.astype(np.uint8)+b.astype(np.uint8)).ravel(),minlength=4)
    assert (int(tp),int(fp),int(fn)) == (result['tp'],result['fp'],result['fn'])
    return dict(**result,tn=int(tn),average_precision=float(average_precision_score(truth.ravel(),score.ravel())))


def score_task(job, run, bank, config, parent_summary):
    name = job['name']; spec = read(job['evaluation']); truth = np.asarray(Image.open(spec['gt_mask']))
    truths = dict(retained_gt_gt127=truth>127,ludvig_gt_maximum=(truth//truth.max()).astype(bool))
    wh = [truth.shape[1],truth.shape[0]]; rows = {}
    for arm in config['arms']:
        p = bank.read(run,name+'_'+arm,1).numpy().reshape(756,1008)
        z = bank.read(run,name+'_'+arm+'_logits',1).numpy().reshape(756,1008)
        p32 = cv2.resize(p,tuple(wh),interpolation=cv2.INTER_LINEAR)
        p64,stable = stable_probability_readout(z,wh)
        out = dict(max_abs_probability_delta=float(np.max(np.abs(p64-p32))),
            float64_hard_mask_changes=int(np.count_nonzero((p64>=.5)!=(p32>=.5))),
            stable_hard_mask_changes=int(np.count_nonzero((stable>=0)!=(p32>=.5))),tracks={})
        for track,g in truths.items():
            native_gt = cv2.resize(g.astype(np.uint8),(1008,756),interpolation=cv2.INTER_NEAREST).astype(bool)
            variants = {v:metrics(x,g,t) for v,x,t in [('original_float32',p32,.5),('float64',p64,.5),('stable_mixture_rank',stable,0)]}
            before = parent_summary['rows'][name][arm][track]
            assert all(variants['original_float32'][k]==before[k] for k in ['tp','fp','fn','tn'])
            assert abs(variants['original_float32']['average_precision']-before['average_precision'])<1e-12
            out['tracks'][track] = dict(benchmark_grid=variants,
                saturated_original=saturation_counts(p32,g),saturated_float64=saturation_counts(p64,g),
                native_grid_diagnostic_ap={v:float(average_precision_score(native_gt.ravel(),x.ravel()))
                    for v,x in [('original_float32',p),('float64',expit(z.astype(np.float64))),('raw_logits',z)]},
                native_saturated=saturation_counts(p,native_gt))
        rows[arm] = out
        print('scored',name,arm,[(v,round(r['average_precision']*100,5)) for v,r in out['tracks']['retained_gt_gt127']['benchmark_grid'].items()],flush=True)
        del p32,p64,stable;gc.collect()
    write(run,name+'_evaluation.json',rows)
    return rows


def main():
    os.chdir(REPO);torch.set_num_threads(4);cv2.setNumThreads(4)
    config = read(CONFIG); parent = REPO/config['study_root']/config['parent_run'];pm=read(parent/'manifest.json')
    for p,h in pm['worker_source_hashes'].items(): assert sha(p)==h,p
    for p,h in pm['source_hashes'].items(): assert sha(p)==h,p
    dataset=read(MANIFEST);jobs=pm['jobs']
    ancestor=read(STUDY/pm['config']['parent_run']/'manifest.json')
    assert sha(MANIFEST)==ancestor['dataset_manifest_sha256']
    for p,h in [('image_checkpoint','checkpoint_sha256'),('dino_checkpoint','dino_checkpoint_sha256')]:
        assert sha(ancestor['config'][p])==ancestor[h]
    gt={str(Path(read(j['evaluation'])['gt_mask']).resolve()) for j in jobs}
    images={str(Path(f['rgb_path']).resolve()) for s in dataset['scenes'] for f in s['frames'] if f['frame_id'] in s['evaluation_frame_ids']}
    access=dict(gt=False,rgb=False)
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=str(Path(os.fsdecode(args[0])).resolve())
            if p in gt and not access['gt']: raise RuntimeError('score audit GT before complete seal')
            if p in images and not access['rgb']: raise RuntimeError('undeclared target RGB access')
    sys.addaudithook(guard)
    run=BoundedExperiment(REPO/config['study_root'],config['run'],config['storage_policy']);run.preflight(config['maximum_new_durable_bytes'])
    bank=RamPredictionBank(limit=config['maximum_volatile_prediction_bytes']);reports=[]
    sources=[__file__,CONFIG,OBSERVER,'radio_gs/evaluation/response_score_precision.py','tests/test_response_score_precision.py','docs/adr/0040-response-score-precision-audit.md']
    write(run,'manifest.json',dict(config=config,jobs=jobs,parent_manifest_sha256=sha(parent/'manifest.json'),
        parent_summary_sha256=sha(parent/'summary.json'),source_hashes={str(p):sha(p) for p in sources},
        external_interface_hashes={p:sha(p) for p in EXTERNAL},volatile_directory=str(bank.root),diagnostic_only=True))
    write(run,'source_snapshot.json',{str(p):dict(sha256=sha(p),text=Path(p).read_text()) for p in sources})
    try:
        for job in jobs:
            reports.append(replay(job,run,bank,config,parent,pm,dataset,access));gc.collect()
        write(run,'predictions_complete.json',dict(jobs=8,probabilities=24,logit_maps=24,target_gt_read=False,
            report_hashes={r['name']:sha(run.root/(r['name']+'_prediction.json')) for r in reports}))
        access['gt']=True;print('all24 exact probability replays + 24 logit maps sealed; opening GT',flush=True)
        previous=read(parent/'summary.json');rows={j['name']:score_task(j,run,bank,config,previous) for j in jobs}
        macro={track:{arm:{v:{m:float(np.mean([r[arm]['tracks'][track]['benchmark_grid'][v][m] for r in rows.values()]))
            for m in ['iou','average_precision']} for v in ['original_float32','float64','stable_mixture_rank']}
            for arm in config['arms']} for track in ['retained_gt_gt127','ludvig_gt_maximum']}
        write(run,'summary.json',dict(complete=True,diagnostic_only=True,rows=rows,macro=macro,
            gt_hashes={p:sha(p) for p in gt},prediction_seal_sha256=sha(run.root/'predictions_complete.json'),
            source_hashes_unchanged=all(sha(p)==sha_record for p,sha_record in read(run.root/'manifest.json')['source_hashes'].items()),
            actual_3d_rerun=False,default_changed=False,independent_test=False,
            other_benchmarks_freshly_evaluated=False,durable_dense_prediction_bytes=0))
        print(json.dumps(macro,indent=2),flush=True)
    except Exception as error:
        write(run,'failure.json',dict(type=type(error).__name__,message=str(error),completed_predictions=len(reports),target_gt_opened=access['gt']));raise
    finally:
        bank.close()
        write(run,'cleanup.json',dict(bank_released=not bank.root.exists(),scratch_released=all(not Path(r['scratch_directory']).exists() for r in reports)))


if __name__=='__main__':main()
