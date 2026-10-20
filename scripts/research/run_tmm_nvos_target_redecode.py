"""N16: matched full-frame and local decoding after complete query transport."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.querying.coverage_evidence import coverage_evidence
from radio_gs.querying.image_conditioned_response import lift_image_response
from scripts.research.run_tmm_nvos_instance_extent import contributions
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from scripts.research.audit_tmm_protocol_correctness import REPO, SYSTEM, binary_counts, read, load
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.tmm_ram_predictions import RamPredictionBank
from scripts.research.run_tmm_support_agreement import array_sha
from scripts.research.run_tmm_nvos_image_response import camera_args, render_responses, MANIFEST
from scripts.research.run_tmm_nvos_signed_affinity import WORKER_SOURCES as AFFINITY_SOURCES
from scripts.research.run_tmm_nvos_memory_response import WORKER_SOURCES as MEMORY_SOURCES
WORKER_SOURCES = list(dict.fromkeys(AFFINITY_SOURCES + MEMORY_SOURCES + [
    "scripts/research/decode_tmm_nvos_reference_completion.py", "radio_gs/querying/reference_completion.py",
    "scripts/research/decode_tmm_nvos_reference_admission.py", "radio_gs/querying/reference_admission.py"]))

WORKER_SOURCES += ['scripts/research/decode_tmm_nvos_target_redecode.py', 'radio_gs/querying/response_crop.py']

CONFIG = 'configs/research/nvos_target_redecode.json'
POLICIES = ['quality', 'full', 'crop']


def write(run, name, value):
    data = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
    return run._write(name, lambda f: f.write(data), limit=len(data))


def binary(path):
    x = np.asarray(Image.open(path))
    return torch.from_numpy(np.any(x > 127, axis=2) if x.ndim == 3 else x > 127)



@torch.inference_mode()
def predict(job, run, bank, config, parent, pm, dataset, access):
    started = time.monotonic(); name = job['name']
    scene, query, spec = [read(job[k]) for k in ['scene', 'queries', 'evaluation']]
    previous = read(parent / (name + '_prediction.json'))
    access['rgb'] = True
    for p, h in previous['input_hashes'].items(): assert sha(p) == h, p
    access['rgb'] = False
    data = load(SYSTEM / 'nvos' / name / 'query/typed_readout.pt')
    old, observed = data['scores'].float(), data['semantic_observed'].bool()
    binding = data['semantic_geometry']
    geo = FrozenGaussianGeometry.load(binding['path'], expected_sha256=binding['sha256'], device='cuda')
    gp = MogeGeometryPrior(scene['camera_report'], scene['camera_report_sha256'])
    target, reference = spec['target_frame'], query['reference_frame']
    assert target not in scene['source_frames'] and target not in scene['tracking_frames']
    fit = read(Path(binding['path']).parent / 'fit.json')
    assert target not in fit['source_frames'] and target not in fit['validation_frames']
    args = camera_args(geo, gp, target, config['native_wh'])
    response = render_responses(args, [old], observed)[..., 0]
    retained = load(SYSTEM / 'nvos' / name / 'evaluation/response.pt')['response']
    error = float((response - retained).abs().max())
    assert error < 1e-4 and torch.equal(response >= .5, retained >= .5)
    ds = next(s for s in dataset['scenes'] if s['scene_id'] == name)
    frame = next(f for f in ds['frames'] if int(f['canonical_index']) == target)
    assert frame['frame_id'] in ds['evaluation_frame_ids']
    assert Path(frame['gt_mask_path']).resolve() == Path(spec['gt_mask']).resolve()
    assert not query['reference_mask_is_gt']
    for sign in ['positive', 'negative']:
        assert sha(ds['prompt'][sign + '_path']) == dataset['protocol']['prompt_asset_sha256'][name][sign]
    values = {'v10': response}
    pc = pm['config']
    with tempfile.TemporaryDirectory(prefix='radiogs_target_redecode_', dir='/dev/shm') as tmp:
        root = Path(tmp); np.save(root / 'prior.npy', response.numpy(), allow_pickle=False)
        access['rgb'] = True
        Image.open(frame['rgb_path']).convert('RGB').resize(tuple(config['native_wh']), Image.Resampling.BILINEAR).save(root / 'rgb.png')
        access['rgb'] = False
        (root / 'request.json').write_text(json.dumps(dict(checkpoint=pc['image_checkpoint'],
            gpu_memory_fraction=config['gpu_memory_fraction'], source_paths=WORKER_SOURCES,
            reference_mask=query['reference_mask'], reference_rgb=query['reference_rgb'],
            positive_scribble=query['positive_scribble'], negative_scribble=ds['prompt']['negative_path'],
            dino_repository=pc['dino_repository'], dino_checkpoint=pc['dino_checkpoint'],
            target_rgb=frame['rgb_path'], points_per_sign=config['points_per_sign'], context_fraction=config['context_fraction'])))
        gc.collect(); torch.cuda.empty_cache()
        subprocess.run(['/root/miniconda3/envs/sam3/bin/python', 'scripts/research/decode_tmm_nvos_target_redecode.py', '--request', str(root)], check=True)
        worker = read(root / 'result.json')
        assert not worker['target_gt_available'] and worker['sam_eval'] and worker['dino_eval'] and worker['negative_scribble_used']
        registered = read(run.root / 'manifest.json')['worker_source_hashes']
        assert all(worker['source_hashes'][p] == registered[p] == sha(p) for p in WORKER_SOURCES)
        candidate = read(REPO/config['study_root']/config['candidate_run']/(name+'_prediction.json'))
        assert worker['choices']==candidate['decoder']['choices']
        for policy in POLICIES:
            values[policy+'_image']=torch.from_numpy(np.load(root/(policy+'_image.npy'),allow_pickle=False))
        replay={a:array_sha(values[a].numpy().reshape(-1,1)) for a in ['v10','quality_image']}
        assert all(replay[a]==candidate['predictions'][a]['sha256'] for a in replay),'N14 image replay changed'
        assert all(x['outside_changed']==0 for x in worker['refinement'].values())
        transient_bytes=sum(p.stat().st_size for p in root.iterdir());assert transient_bytes<64*2**20
    w, h = config['native_wh']; hit, target_error = contributions(geo, args, config)
    pid, gid = hit['pixel_ids'].long(), hit['gaussian_ids'].long()
    weight = hit['weights'].float() * observed.cuda()[gid]
    mass = torch.zeros(h*w, device='cuda').scatter_add_(0, pid, weight)
    valid = (mass / hit['rendered_alpha'].flatten().clamp_min(1e-8) > .5) & (mass > 1e-8)
    op = torch.sparse_coo_tensor(torch.stack([pid, gid]), weight / mass[pid].clamp_min(1e-8), (h*w, len(old))).coalesce().cpu()
    valid = valid.cpu()[:, None]
    check = torch.sparse.mm(op, old); check[~valid] = 0
    operator_error = float((check.reshape(h, w) - response).abs().max()); assert operator_error < 1e-4
    del hit, pid, gid, weight, mass; torch.cuda.empty_cache()
    fields, lifting = {}, {}
    for policy in POLICIES:
        y = values[policy + '_image'].reshape(-1, 1); c = (2*y-1).abs() * valid
        arm = policy + '_native'
        fields[arm], lifting[arm] = lift_image_response(old, op, y, c, config['solver_steps'])
        assert lifting[arm]['unsupported_changed'] == 0 and not torch.count_nonzero(fields[arm][~observed])
        values[arm] = render_responses(args, [fields[arm]], observed)[..., 0]
    del op
    source = []; sh, sw = config['source_hw']; tracks = load(job['reference_tracks'])['frames']; field_arms = ['v10', *fields]
    for f in scene['source_frames']:
        r = tracks.get(f)
        if r is None or r['object_logit'] <= 0: continue
        y, c = coverage_evidence(r['logits'].float(), (sh, sw))
        image = render_responses(camera_args(geo, gp, f, (sw, sh)), [old, *fields.values()], observed).reshape(-1, len(field_arms))
        e = (image - y).abs(); loss = torch.where(e <= .25, .5 * e.square(), .25 * (e - .125))
        source.append(dict(frame=f, confidence_weighted_mean_huber={a: float((loss[:, i:i+1] * c).sum() / c.sum().clamp_min(1e-8)) for i, a in enumerate(field_arms)}))
    predictions = {a: bank.add(run, name + '_' + a, values[a].reshape(-1, 1)) for a in config['arms']}
    report = dict(name=name, predictions=predictions, parent_report_sha256=sha(parent / (name + '_prediction.json')),
        candidate_report_sha256=sha(REPO / config['study_root'] / config['candidate_run'] / (name + '_prediction.json')),
        parent_arrays=replay, decoder=worker, lifting=lifting, source_teacher_checks=source,
        transient_directory=str(root), transient_bytes=transient_bytes,
        target_gt_read=False, seconds=time.monotonic()-started,
        implementation_checks=dict(v10_binary_exact=True, v10_render_max_error=error, target_alpha_max_error=target_error, normalized_operator_max_error=operator_error))
    write(run, name + '_prediction.json', report)
    print('sealed', name, 'seven arms', round(report['seconds'], 1), 'seconds', flush=True)
    return report


def main():
    os.chdir(REPO); config = read(CONFIG); parent = STUDY / config['parent_run']; pm = read(parent / 'manifest.json')
    jobs = pm['jobs']; dataset = read(MANIFEST); torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    gt = {str(Path(read(j['evaluation'])['gt_mask']).resolve()) for j in jobs}
    images = {str(Path(f['rgb_path']).resolve()) for s in dataset['scenes'] for f in s['frames'] if f['frame_id'] in s['evaluation_frame_ids']}
    access = dict(gt=False, rgb=False)
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = str(Path(os.fsdecode(args[0])).resolve())
            if p in gt and not access['gt']: raise RuntimeError('N16 GT before full prediction seal')
            if p in images and not access['rgb']: raise RuntimeError('undeclared target RGB access')
    sys.addaudithook(guard)
    run = BoundedExperiment(REPO / config['study_root'], config['run'], config['storage_policy']); run.preflight(config['maximum_new_durable_bytes'])
    bank = RamPredictionBank(limit=config['maximum_volatile_prediction_bytes'])
    sources = [__file__, CONFIG, 'radio_gs/querying/response_crop.py',
        'scripts/research/decode_tmm_nvos_target_redecode.py', 'docs/adr/0039-query-local-target-redecoding.md']
    inherited = read(parent / 'source_snapshot.json')
    for p, v in inherited.items(): assert sha(p) == v['sha256'], p
    assert sha(MANIFEST) == pm['dataset_manifest_sha256']
    for k, h in [('image_checkpoint', 'checkpoint_sha256'), ('dino_checkpoint', 'dino_checkpoint_sha256')]: assert sha(pm['config'][k]) == pm[h]
    write(run, 'manifest.json', dict(config=config, jobs=jobs, source_hashes={str(p): sha(p) for p in sources},
        inherited_source_snapshot_sha256=sha(parent / 'source_snapshot.json'), parent_manifest_sha256=sha(parent / 'manifest.json'),
        volatile_prediction_directory=str(bank.root), torch_version=torch.__version__,
        worker_source_hashes={p: sha(p) for p in WORKER_SOURCES},
        inherited_memory_snapshot_sha256=sha(STUDY / 'nvos_memory_response_v2/source_snapshot.json'),
        inherited_instance_snapshot_sha256=sha(STUDY / 'nvos_instance_extent_v2/source_snapshot.json'),
        inherited_completion_snapshot_sha256=sha(REPO / config['study_root'] / config['candidate_run'] / 'source_snapshot.json'),
        candidate_manifest_sha256=sha(REPO / config['study_root'] / config['candidate_run'] / 'manifest.json')))
    write(run, 'source_snapshot.json', {str(p): dict(sha256=sha(p), text=Path(p).read_text()) for p in sources})
    for parent_run in ['nvos_memory_response_v2', 'nvos_instance_extent_v2']:
        for path, entry in read(STUDY / parent_run / 'source_snapshot.json').items():
            assert sha(path) == entry['sha256'], path
    for path, entry in read(REPO / config['study_root'] / config['candidate_run'] / 'source_snapshot.json').items():
        assert sha(path) == entry['sha256'], path
    write(run, 'completion_lineage.json', {r:sha(REPO/config['study_root']/r/'source_snapshot.json') for r in ['nvos_reference_completion_v1',config['candidate_run']]})
    reports = []
    try:
        for job in jobs:
            reports.append(predict(job, run, bank, config, parent, pm, dataset, access)); gc.collect(); torch.cuda.empty_cache()
        write(run, 'predictions_complete.json', dict(jobs=8, images=56, target_gt_read=False, threshold=.5,
            report_hashes={r['name']: sha(run.root / (r['name'] + '_prediction.json')) for r in reports},
            predictions={r['name']: r['predictions'] for r in reports}))
        access['gt'] = True; previous = read(REPO/config['study_root']/config['candidate_run']/'summary.json'); rows, gt_hashes, transitions = {}, {}, {}
        print('all56 sealed; opening GT', flush=True)
        for job in jobs:
            name = job['name']; spec = read(job['evaluation']); truth = np.asarray(Image.open(spec['gt_mask']))
            assert truth.ndim == 2 and truth.max() > 0
            truths = dict(retained_gt_gt127=truth > 127, ludvig_gt_maximum=(truth // truth.max()).astype(bool))
            gt_hashes[spec['gt_mask']] = sha(spec['gt_mask']); rows[name] = {}; masks = {}
            for arm in config['arms']:
                x = bank.read(run, name + '_' + arm, 1).reshape(spec['raster_wh'][1], spec['raster_wh'][0]).numpy()
                x = cv2.resize(x, (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_LINEAR)
                assert np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1 + 1e-6
                b = x >= .5; rows[name][arm] = {}
                if arm.endswith('_native'): masks[arm] = b
                for track, g in truths.items():
                    r = binary_counts(b, g); tn, fp, fn, tp = np.bincount((2*g.astype(np.uint8) + b.astype(np.uint8)).ravel(), minlength=4).tolist()
                    assert (tp, fp, fn) == (r['tp'], r['fp'], r['fn'])
                    value = dict(**r, tn=tn, accuracy=(tp+tn)/g.size, independent_confusion_exact=True)
                    if arm in ['v10','quality_image','quality_native']:
                        before=previous['rows'][name][arm][track]
                        assert all(value[k]==before[k] for k in ['tp','fp','fn','tn']),(name,arm,'N14 control drift')
                    if arm in ['v10','quality_image']:
                        value['average_precision']=previous['rows'][name][arm][track]['average_precision']
                    else:
                        from sklearn.metrics import average_precision_score
                        value['average_precision']=float(average_precision_score(g.ravel(),x.ravel()))
                    rows[name][arm][track] = value
            old = masks['quality_native']; g = truths['retained_gt_gt127']; transitions[name] = {}
            for arm, b in masks.items():
                transitions[name][arm] = dict(recovered_foreground=int((b & ~old & g).sum()), lost_foreground=int((~b & old & g).sum()),
                    removed_false_positive=int((~b & old & ~g).sum()), added_false_positive=int((b & ~old & ~g).sum()))
            write(run, name + '_evaluation.json', rows[name]); print('scored', name, flush=True)
        macro = {t: {a: {m: float(np.mean([r[a][t][m] for r in rows.values()])) for m in config['metrics']} for a in config['arms']} for t in config['gt_tracks']}
        write(run, 'summary.json', dict(complete=True, rows=rows, macro=macro, gt_hashes=gt_hashes, primary_arm=config['primary_arm'],
            prediction_seal_sha256=sha(run.root / 'predictions_complete.json'), parent_summary_sha256=sha(parent / 'summary.json'),
            transitions_diagnostic_only=transitions, target_rgb_used=True, negative_scribble_used=True,
            target_gt_used_for_prediction=False, default_changed=False, per_task_winners=False, previous_candidate=config['candidate_run'],
            candidate_summary_sha256=sha(REPO/config['study_root']/config['candidate_run']/'summary.json'),
            previous_candidate_summary_sha256=sha(REPO/config['study_root']/config['candidate_run']/'summary.json'), independent_test=False,
            average_precision_measured=True, durable_dense_prediction_bytes=0, other_benchmarks=previous['other_benchmarks'], other_benchmarks_freshly_evaluated=False))
        print(json.dumps(macro, indent=2), flush=True)
    except Exception as error:
        write(run, 'failure.json', dict(type=type(error).__name__, message=str(error), completed_predictions=len(reports), target_gt_opened=access['gt'])); raise
    finally: bank.close()


if __name__ == '__main__': main()
