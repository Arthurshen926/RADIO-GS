"""N15: completed-reference multi-view evidence and native target fusion."""
import gc
import hashlib
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
from radio_gs.querying.multiview_response import fit_multiview_response
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
    "scripts/research/decode_tmm_nvos_reference_admission.py", "radio_gs/querying/reference_admission.py",
    "scripts/research/decode_tmm_nvos_multiview_completion.py"]))

CONFIG = 'configs/research/nvos_multiview_completion.json'
POLICIES = ['feasible', 'quality']


def write(run, name, value):
    data = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
    return run._write(name, lambda f: f.write(data), limit=len(data))


def binary(path):
    x = np.asarray(Image.open(path))
    return torch.from_numpy(np.any(x > 127, axis=2) if x.ndim == 3 else x > 127)



@torch.inference_mode()
def make_operator(geo, gp, frame, wh, old, observed, config):
    args = camera_args(geo, gp, frame, wh)
    hit, error = contributions(geo, args, config)
    w, h = wh
    pid, gid = hit['pixel_ids'].long(), hit['gaussian_ids'].long()
    weight = hit['weights'].float() * observed.cuda()[gid]
    mass = torch.zeros(h*w, device='cuda').scatter_add_(0, pid, weight)
    valid = ((mass / hit['rendered_alpha'].flatten().clamp_min(1e-8) > .5) & (mass > 1e-8)).cpu()[:, None]
    op = torch.sparse_coo_tensor(torch.stack([pid, gid]), weight/mass[pid].clamp_min(1e-8), (h*w, len(old))).coalesce().cpu()
    check = torch.sparse.mm(op, old); check[~valid] = 0
    renderer = render_responses(args, [old], observed)[..., 0]
    delta = float((check.reshape(h,w)-renderer).abs().max()); assert delta < 1e-4
    return op, valid, dict(alpha_max_error=error, normalized_operator_max_error=delta, sparse_hits=op._nnz())


@torch.inference_mode()
def predict(job, run, bank, config, parent, pm, dataset, access):
    started = time.monotonic(); name = job['name']
    scene, query, spec = [read(job[k]) for k in ['scene','queries','evaluation']]
    previous = read(parent/(name+'_prediction.json'))
    candidate_root = REPO/config['study_root']/config['candidate_run']
    candidate = read(candidate_root/(name+'_prediction.json'))
    access['rgb'] = True
    for p,h in previous['input_hashes'].items(): assert sha(p)==h,p
    access['rgb'] = False
    data = load(SYSTEM/'nvos'/name/'query/typed_readout.pt')
    old, observed = data['scores'].float(), data['semantic_observed'].bool()
    binding = data['semantic_geometry']
    geo = FrozenGaussianGeometry.load(binding['path'], expected_sha256=binding['sha256'], device='cuda')
    gp = MogeGeometryPrior(scene['camera_report'], scene['camera_report_sha256'])
    target, reference = spec['target_frame'], query['reference_frame']
    assert target not in scene['source_frames'] and target not in scene['tracking_frames']
    assert reference in scene['source_frames'] and not query['reference_mask_is_gt']
    fit = read(Path(binding['path']).parent/'fit.json')
    assert target not in fit['source_frames'] and target not in fit['validation_frames']
    args = camera_args(geo,gp,target,config['native_wh'])
    response = render_responses(args,[old],observed)[...,0]
    retained = load(SYSTEM/'nvos'/name/'evaluation/response.pt')['response']
    error = float((response-retained).abs().max()); assert error<1e-4 and torch.equal(response>=.5,retained>=.5)
    ds = next(s for s in dataset['scenes'] if s['scene_id']==name)
    frame = next(f for f in ds['frames'] if int(f['canonical_index'])==target)
    assert frame['frame_id'] in ds['evaluation_frame_ids'] and Path(frame['gt_mask_path']).resolve()==Path(spec['gt_mask']).resolve()
    for sign in ['positive','negative']:
        assert sha(ds['prompt'][sign+'_path'])==dataset['protocol']['prompt_asset_sha256'][name][sign]
    assert Path(scene['source_rgb'][str(reference)]).resolve()==Path(query['reference_rgb']).resolve()
    source_hashes = {p:sha(p) for p in scene['source_rgb'].values()}
    values={'v10':response}; pc=pm['config']; source_terms=[]; operator_checks=[]; source_records=[]
    with tempfile.TemporaryDirectory(prefix='radiogs_multiview_',dir='/dev/shm') as tmp:
        root=Path(tmp); np.save(root/'prior.npy',response.numpy(),allow_pickle=False)
        request=dict(checkpoint=pc['image_checkpoint'], gpu_memory_fraction=config['gpu_memory_fraction'],
            source_paths=WORKER_SOURCES, reference_mask=query['reference_mask'], reference_rgb=query['reference_rgb'],
            positive_scribble=query['positive_scribble'], negative_scribble=ds['prompt']['negative_path'],
            dino_repository=pc['dino_repository'],dino_checkpoint=pc['dino_checkpoint'],target_rgb=frame['rgb_path'],
            points_per_sign=config['points_per_sign'],source_frames=scene['source_frames'],source_rgb=scene['source_rgb'],
            reference_frame=reference,source_hw=config['source_hw'],maximum_source_evidence_bytes=config['maximum_source_evidence_bytes'])
        (root/'request.json').write_text(json.dumps(request)); gc.collect(); torch.cuda.empty_cache()
        subprocess.run(['/root/miniconda3/envs/sam3/bin/python','scripts/research/decode_tmm_nvos_multiview_completion.py','--request',str(root)],check=True)
        worker=read(root/'result.json'); registered=read(run.root/'manifest.json')['worker_source_hashes']
        assert worker['source_evidence_before_target_rgb'] and not worker['target_gt_available']
        assert all(worker['source_hashes'][p]==registered[p]==sha(p) for p in WORKER_SOURCES)
        assert worker['choices']==candidate['decoder']['choices']
        chosen=worker['choices']['quality']
        completion=read(REPO/config['study_root']/'nvos_reference_completion_v1'/(name+'_prediction.json'))
        assert worker['selected_reference_sha256']==completion['decoder']['decoder'][chosen]['reference_sha256']
        values['quality_image']=torch.from_numpy(np.load(root/'quality_image.npy',allow_pickle=False))
        replay={a:array_sha(values[a].numpy().reshape(-1,1)) for a in ['v10','quality_image']}
        assert all(replay[a]==candidate['predictions'][a]['sha256'] for a in replay),'N14 replay drift'
        transient_bytes=sum(p.stat().st_size for p in root.iterdir()); assert transient_bytes<config['maximum_source_evidence_bytes']
        op, valid, target_check=make_operator(geo,gp,target,config['native_wh'],old,observed,config)
        y=values['quality_image'].reshape(-1,1); c=(2*y-1).abs()*valid
        incumbent, incumbent_fit=lift_image_response(old,op,y,c,config['solver_steps'])
        values['quality_native']=render_responses(args,[incumbent],observed)[...,0]
        target_term=(op,y.expand(-1,3),torch.cat([torch.zeros_like(c),torch.zeros_like(c),c],1))
        sparse_bytes=op._nnz()*20
        sh,sw=config['source_hw']
        for f in scene['source_frames']:
            a,v,check=make_operator(geo,gp,f,(sw,sh),old,observed,config)
            sparse_bytes+=a._nnz()*20; assert sparse_bytes<config['maximum_sparse_operator_bytes']
            evidence=[]
            for branch in ['legacy','completed']:
                x=np.load(root/f'{branch}_{f}.npy',allow_pickle=False)
                assert hashlib.sha256(x.tobytes()).hexdigest()==worker['sources'][str(f)][branch]['sha256']
                assert x.shape==(sh*sw,2) and np.isfinite(x).all() and x.min()>=0 and x.max()<=1
                evidence.append(torch.from_numpy(x))
            ly,lc=evidence[0].split(1,dim=1); ny,nc=evidence[1].split(1,dim=1)
            ys=torch.cat([ly,ny,ny],1); cs=torch.cat([lc,nc,nc],1)*v
            source_terms.append((a,ys,cs)); operator_checks.append(dict(frame=f,**check))
        print('source operators ready',name,len(source_terms),'views',round(sparse_bytes/2**30,2),'GiB',flush=True)
    field, lifting=fit_multiview_response(old.expand(-1,3).clone(),source_terms,target_term,config['solver_steps'])
    assert lifting['target_confidence_mass'][:2]==[0.,0.] and lifting['unsupported_changed']==0
    assert not torch.count_nonzero(field[~observed])
    fields={a:field[:,i:i+1] for i,a in enumerate(['legacy_source_native','completed_source_native','completed_joint_native'])}
    rendered=render_responses(args,list(fields.values()),observed)
    values.update({a:rendered[...,i] for i,a in enumerate(fields)})
    # Consistency is measured on both original and completed source evidence;
    # neither pseudo-mask is treated as ground truth.
    source_fields=torch.cat([old,incumbent,field],1)
    field_names=['v10','quality_native',*fields]
    for f,(a,ys,cs) in zip(scene['source_frames'],source_terms):
        pred=torch.sparse.mm(a.to_sparse_csr(),source_fields)
        record=dict(frame=f)
        for branch,i in [('legacy',0),('completed',1)]:
            e=(pred-ys[:,i:i+1]).abs(); loss=torch.where(e<=.25,.5*e.square(),.25*(e-.125))
            record[branch]={n:float((loss[:,k:k+1]*cs[:,i:i+1]).sum()/cs[:,i:i+1].sum().clamp_min(1e-8)) for k,n in enumerate(field_names)}
        source_records.append(record)
    predictions={a:bank.add(run,name+'_'+a,values[a].reshape(-1,1)) for a in config['arms']}
    report=dict(name=name,predictions=predictions,parent_report_sha256=sha(parent/(name+'_prediction.json')),
        candidate_report_sha256=sha(candidate_root/(name+'_prediction.json')),parent_arrays=replay,
        source_rgb_hashes=source_hashes,decoder=worker,lifting=lifting,incumbent_fit=incumbent_fit,
        source_consistency=source_records,source_operator_checks=operator_checks,
        source_frames=scene['source_frames'],target_frame=target,target_gt_read=False,
        transient_source_directory=str(root),transient_source_bytes=transient_bytes,sparse_operator_bytes=sparse_bytes,
        seconds=time.monotonic()-started,implementation_checks=dict(v10_binary_exact=True,v10_render_max_error=error,**target_check))
    write(run,name+'_prediction.json',report)
    print('sealed',name,'six arms',round(report['seconds'],1),'seconds',flush=True)
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
            if p in gt and not access['gt']: raise RuntimeError('N15 GT before full prediction seal')
            if p in images and not access['rgb']: raise RuntimeError('undeclared target RGB access')
    sys.addaudithook(guard)
    run = BoundedExperiment(REPO / config['study_root'], config['run'], config['storage_policy']); run.preflight(config['maximum_new_durable_bytes'])
    bank = RamPredictionBank(limit=config['maximum_volatile_prediction_bytes'])
    sources = [__file__, CONFIG, 'radio_gs/querying/multiview_response.py',
        'scripts/research/decode_tmm_nvos_multiview_completion.py', 'docs/adr/0038-completed-reference-multiview-fusion.md']
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
    write(run, 'completion_lineage.json', {r: sha(REPO/config['study_root']/r/'source_snapshot.json') for r in ['nvos_reference_completion_v1',config['candidate_run']]})
    reports = []
    try:
        for job in jobs:
            reports.append(predict(job, run, bank, config, parent, pm, dataset, access)); gc.collect(); torch.cuda.empty_cache()
        write(run, 'predictions_complete.json', dict(jobs=8, images=48, target_gt_read=False, threshold=.5,
            report_hashes={r['name']: sha(run.root / (r['name'] + '_prediction.json')) for r in reports},
            predictions={r['name']: r['predictions'] for r in reports}))
        access['gt'] = True; previous = read(REPO/config['study_root']/config['candidate_run']/'summary.json'); memory = read(STUDY / 'nvos_memory_response_v2/summary.json'); rows, gt_hashes, transitions = {}, {}, {}
        print('all48 sealed; opening GT', flush=True)
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
                    if arm in ['v10', 'quality_image', 'quality_native']:
                        before = previous['rows'][name][arm][track]; assert all(value[k] == before[k] for k in value), (name, arm, 'control drift')
                    if arm == 'legacy_memory_image':
                        before = memory['rows'][name]['pair_memory'][track]
                        assert all(value[k] == before[k] for k in ['tp','fp','fn']), (name, 'legacy memory changed')
                    if arm in ['v10', 'quality_image']:
                        value['average_precision'] = previous['rows'][name][arm][track]['average_precision']
                    else:
                        from sklearn.metrics import average_precision_score
                        value['average_precision'] = float(average_precision_score(g.ravel(), x.ravel()))
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
