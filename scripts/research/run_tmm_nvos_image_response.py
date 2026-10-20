"""N04: paired RGB access, response prompting and actual 3D writeback, all eight."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from gsplat import rasterization
from sklearn.metrics import average_precision_score, confusion_matrix

from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.querying.coverage_evidence import coverage_evidence
from radio_gs.querying.image_conditioned_response import response_prompt, lift_image_response
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from scripts.research.audit_tmm_protocol_correctness import REPO, SYSTEM, binary_counts, read, load
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.tmm_ram_predictions import RamPredictionBank

CONFIG = 'configs/research/nvos_image_response.json'
MANIFEST = '/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json'
DECODER_SOURCES = ['scripts/research/decode_tmm_nvos_image_response.py',
    'radio_gs/querying/image_conditioned_response.py', 'radio_gs/scripts/build_sam3_foundation_cache.py',
    '/root/external/sam3/sam3/model_builder.py', '/root/external/sam3/sam3/model/sam3_image.py',
    '/root/external/sam3/sam3/model/sam3_image_processor.py',
    '/root/external/sam3/sam3/model/sam1_task_predictor.py']


def camera_args(geo, gp, frame, wh):
    width, height = wh
    cam = gp.cameras[frame]
    k = cam.intrinsic.cuda().clone()
    k[0] *= width / cam.width
    k[1] *= height / cam.height
    return dict(means=geo.xyz, quats=geo.quaternions, scales=geo.scales,
                opacities=geo.opacities.flatten(), viewmats=cam.world_to_camera.cuda()[None],
                Ks=k[None], width=width, height=height,
                near_plane=.001 * gp.scale, far_plane=100 * gp.scale, packed=False)


def render_responses(args, values, observed):
    colors = torch.cat([v.cuda() for v in values] + [observed.cuda().float()[:, None]], dim=1)
    image, alpha, _ = rasterization(**args, colors=colors)
    mass = image[0, ..., -1]
    valid = (mass / alpha[0, ..., 0].clamp_min(1e-8) > .5) & (alpha[0, ..., 0] > 1e-8)
    return torch.where(valid[..., None], image[0, ..., :-1] / mass.clamp_min(1e-8)[..., None], 0).cpu()


@torch.inference_mode()
def predict(job, run, bank, config, dataset, access):
    started = time.monotonic()
    name = job['name']
    scene, query, spec = read(job['scene']), read(job['queries']), read(job['evaluation'])
    dpath = SYSTEM / 'nvos' / name / 'query/typed_readout.pt'
    data = load(dpath)
    old, observed = data['scores'].float(), data['semantic_observed'].bool()
    binding = data['semantic_geometry']
    geo = FrozenGaussianGeometry.load(binding['path'], expected_sha256=binding['sha256'], device='cuda')
    gp = MogeGeometryPrior(scene['camera_report'], scene['camera_report_sha256'])
    target = spec['target_frame']
    assert target not in scene['source_frames'] and target not in scene['tracking_frames']
    rgb_path = Path(binding['path']).parent / 'rgb_parameters.pt'
    fit_path = rgb_path.parent / 'fit.json'
    fit = read(fit_path)
    assert target not in fit['source_frames'] and target not in fit['validation_frames']
    rgb = load(rgb_path)
    # Bind RGB reconstruction to the exact Gaussian domain, not a similar scene.
    torch.testing.assert_close(rgb['means'], geo.xyz.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(F.normalize(rgb['quats'], dim=-1), geo.quaternions.cpu(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(rgb['scales'].exp(), geo.scales.cpu(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(rgb['opacities'].sigmoid().clamp(1e-6, 1-1e-6), geo.opacities.cpu().flatten(), rtol=1e-6, atol=1e-7)
    args = camera_args(geo, gp, target, spec['raster_wh'])
    response = render_responses(args, [old], observed)[..., 0]
    retained_path = SYSTEM / 'nvos' / name / 'evaluation/response.pt'
    retained = load(retained_path)['response']
    compatibility = float((response - retained).abs().max())
    assert compatibility < 1e-4
    assert torch.equal(response >= .5, retained >= .5)
    rgb_render, _, _ = rasterization(**args, colors=torch.cat([rgb['sh0'], rgb['shN']], 1).cuda(), sh_degree=fit['sh_degree'])
    rendered = Image.fromarray((rgb_render[0].clamp(0, 1) * 255).byte().cpu().numpy())
    del rgb_render, rgb
    values = {'v10': response}
    logits = {}
    ds = next(s for s in dataset['scenes'] if s['scene_id'] == name)
    frame = next(f for f in ds['frames'] if int(f['canonical_index']) == target)
    assert frame['frame_id'] in ds['evaluation_frame_ids']
    assert str(Path(frame['gt_mask_path']).resolve()) == str(Path(spec['gt_mask']).resolve())
    inputs = [job['scene'], job['queries'], job['evaluation'], dpath, retained_path,
              binding['path'], rgb_path, fit_path, scene['camera_report'], job['reference_tracks']]
    # The established gsplat and SAM3 environments use different torch ABIs.
    # Exchange only these small transient arrays; never duplicate checkpoints.
    with tempfile.TemporaryDirectory(prefix='radiogs_image_', dir='/dev/shm') as tmp:
        temporary = Path(tmp)
        rendered.save(temporary / 'render.png')
        np.save(temporary / 'prior.npy', response.numpy(), allow_pickle=False)
        access['target_rgb'] = True
        image = Image.open(frame['rgb_path']).convert('RGB').resize(tuple(spec['raster_wh']), Image.Resampling.BILINEAR)
        image.save(temporary / 'rgb.png')
        access['target_rgb'] = False
        inputs.append(frame['rgb_path'])
        (temporary / 'request.json').write_text(json.dumps(dict(checkpoint=config['image_checkpoint'],
            gpu_memory_fraction=config['gpu_memory_fraction'], source_paths=DECODER_SOURCES)))
        gc.collect()
        torch.cuda.empty_cache()
        subprocess.run(['/root/miniconda3/envs/sam3/bin/python', 'scripts/research/decode_tmm_nvos_image_response.py',
                        '--request', str(temporary)], check=True)
        worker = read(temporary / 'result.json')
        assert not worker['target_gt_available']
        registered_sources = read(run.root / 'manifest.json')['source_hashes']
        assert all(worker['source_hashes'][p] == registered_sources[p] == sha(p) for p in DECODER_SOURCES)
        decoder = worker['decoder']
        for arm in decoder:
            logits[arm] = torch.from_numpy(np.load(temporary / (arm + '.npy'), allow_pickle=False))
            values[arm] = response.clone() if decoder[arm]['fallback_empty_prompt'] else logits[arm].sigmoid()
        assert sum(p.stat().st_size for p in temporary.iterdir()) < 32 * 2**20
    # Target-view operator uses only the ORIGINAL observed Gaussian domain.
    # Audit the exact contribution compositor against independent gsplat output.
    h, w = config['source_hw']
    low_args = camera_args(geo, gp, target, (w, h))
    renderer = SimpleNamespace(scaled_intrinsics=lambda width, height: low_args['Ks'][0],
                               near_plane=low_args['near_plane'], far_plane=low_args['far_plane'])
    hit = rasterize_single_view_contributions(geo, renderer, low_args['viewmats'][0], height=h, width=w, include_depths=False)
    pid, gid, weight = hit['pixel_ids'].long(), hit['gaussian_ids'].long(), hit['weights'].float()
    alpha_error = float((hit['accumulated_alpha'] - hit['rendered_alpha']).abs().max())
    assert alpha_error < 1e-4
    seen_weight = weight * observed.cuda()[gid]
    mass = torch.zeros(h*w, device='cuda').scatter_add_(0, pid, seen_weight)
    keep = (mass / hit['rendered_alpha'].flatten().clamp_min(1e-8) > .5) & (mass > 1e-8)
    op = torch.sparse_coo_tensor(torch.stack([pid, gid]), seen_weight / mass[pid].clamp_min(1e-8), (h*w, len(old))).coalesce().cpu()
    op_image = torch.sparse.mm(op, old).reshape(h, w)
    op_image[~keep.cpu().reshape(h, w)] = 0
    actual = render_responses(low_args, [old], observed)[..., 0]
    operator_error = float((op_image - actual).abs().max())
    assert operator_error < 1e-4
    field_values, lifting = {}, {}
    for arm in ('render_mask', 'rgb_mask'):
        y, confidence = coverage_evidence(logits[arm], (h, w))
        confidence *= keep.cpu()[:, None]
        value, report = lift_image_response(old, op, y, confidence, config['solver_steps'])
        assert not torch.count_nonzero(value[~observed])
        assert report['unsupported_changed'] == 0
        assert report['source_image_loss_after'] <= report['source_image_loss_before'] + 1e-5
        field_values[arm + '_lift'] = value
        lifting[arm + '_lift'] = report
        values[arm + '_lift'] = render_responses(args, [value], observed)[..., 0]
    del hit, op, pid, gid, weight, seen_weight
    # Measure all source views against original teacher evidence, not target GT.
    tracks = load(job['reference_tracks'])['frames']
    source = []
    field_arms = ['v10', *field_values]
    for f in scene['source_frames']:
        record = tracks.get(f)
        if record is None or record['object_logit'] <= 0:
            continue
        ys, ws = coverage_evidence(record['logits'].float(), (h, w))
        image = render_responses(camera_args(geo, gp, f, (w, h)), [old, *field_values.values()], observed).reshape(-1, len(field_arms))
        abs_error = (image - ys).abs()
        loss = torch.where(abs_error <= .25, .5 * abs_error.square(), .25 * (abs_error - .125))
        source.append(dict(frame=f, confidence_weighted_mean_huber={a: float((loss[:, i:i+1] * ws).sum() / ws.sum().clamp_min(1e-8)) for i, a in enumerate(field_arms)}))
    seals = {a: bank.add(run, name + '_' + a, values[a].reshape(-1, 1)) for a in config['arms']}
    # Hash observed target RGB only within its declared access phase.
    access['target_rgb'] = True
    hashes = {str(p): sha(p) for p in inputs}
    access['target_rgb'] = False
    report = dict(job=job, predictions=seals, input_hashes=hashes, decoder=decoder, lifting=lifting,
                  decoder_source_hashes=worker['source_hashes'],
                  source_teacher_checks=source, target_rgb_arms=config['target_rgb_arms'],
                  target_gt_read=False, seconds=time.monotonic()-started,
                  implementation_checks=dict(v10_render_max_error=compatibility, v10_binary_exact=True,
                      rgb_geometry_domain_matches=True, compositor_alpha_max_error=alpha_error,
                      normalized_operator_max_error=operator_error, mask_input_hw=worker['mask_input_hw'],
                      prompt_box=worker['prompt_box'], decoder_torch_version=worker['torch_version']))
    run.json(name + '_prediction.json', report)
    print('sealed', name, 'seven arms', round(report['seconds'], 1), 'seconds', flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run')
    args = p.parse_args()
    os.chdir(REPO)
    torch.set_num_threads(4)
    config = read(CONFIG)
    torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    dataset = read(MANIFEST)
    jobs = read('configs/pipeline/system_v10.json')['jobs']
    nvos = [j for j in jobs if j['benchmark'] == 'nvos']
    forbidden_gt = {str(Path(read(j['evaluation'])['gt_mask']).resolve()) for j in nvos}
    target_images = {str(Path(f['rgb_path']).resolve()) for s in dataset['scenes'] for f in s['frames'] if f['frame_id'] in s['evaluation_frame_ids']}
    access = dict(gt=False, target_rgb=False)
    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = str(Path(os.fsdecode(arguments[0])).resolve())
            if path in forbidden_gt and not access['gt']:
                raise RuntimeError('target GT forbidden before complete prediction seal')
            if path in target_images and not access['target_rgb']:
                raise RuntimeError('target RGB forbidden outside observed-image arm')
    sys.addaudithook(audit)
    run = BoundedExperiment(STUDY, args.run or config['run'], POLICY)
    run.preflight(config['maximum_new_durable_bytes'])
    bank = RamPredictionBank(limit=config['maximum_volatile_prediction_bytes'])
    sources = [__file__, CONFIG, 'scripts/research/decode_tmm_nvos_image_response.py', 'radio_gs/querying/image_conditioned_response.py',
               'radio_gs/querying/coverage_evidence.py', 'radio_gs/querying/response_reconstruction.py',
               'radio_gs/rendering/contribution_compositor.py', 'radio_gs/scripts/build_sam3_foundation_cache.py',
               'scripts/research/tmm_ram_predictions.py', 'radio_gs/training/retrieval_metrics.py',
               'docs/adr/0027-image-conditioned-query-response.md']
    sources = list(dict.fromkeys([*sources, *DECODER_SOURCES]))
    run.json('manifest.json', dict(config=config, jobs=nvos, source_hashes={str(p): sha(p) for p in sources},
        dataset_manifest_sha256=sha(MANIFEST), checkpoint_sha256=sha(config['image_checkpoint']),
        volatile_prediction_directory=str(bank.root), torch_version=torch.__version__,
        other_benchmarks='unchanged references, not fresh inference'))
    run.json('source_snapshot.json', {str(p): dict(sha256=sha(p), text=Path(p).read_text()) for p in sources})
    reports = []
    try:
        for job in nvos:
            reports.append(predict(job, run, bank, config, dataset, access))
            gc.collect()
            torch.cuda.empty_cache()
        run.json('predictions_complete.json', dict(jobs=8, images=56, target_gt_read=False,
            report_hashes={r['job']['name']: sha(run.root / (r['job']['name'] + '_prediction.json')) for r in reports},
            predictions={r['job']['name']: r['predictions'] for r in reports}, threshold=.5))
        gc.collect()
        torch.cuda.empty_cache()
        access['gt'] = True
        rows, gt_hashes = {}, {}
        print('56 predictions sealed; independent scoring now allowed', flush=True)
        for job in nvos:
            name = job['name']
            spec = read(job['evaluation'])
            gt = np.asarray(Image.open(spec['gt_mask']))
            assert gt.ndim == 2 and gt.max() > 0
            gt_hashes[spec['gt_mask']] = sha(spec['gt_mask'])
            truths = dict(retained_gt_gt127=gt > 127, ludvig_gt_maximum=(gt // gt.max()).astype(bool))
            rows[name] = {}
            for arm in config['arms']:
                response = bank.read(run, name + '_' + arm, 1).reshape(spec['raster_wh'][1], spec['raster_wh'][0]).numpy()
                response = cv2.resize(response, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                assert np.isfinite(response).all() and response.min() >= 0 and response.max() <= 1 + 1e-6
                rows[name][arm] = {}
                for track, truth in truths.items():
                    scores = binary_counts(response >= .5, truth)
                    tn, fp, fn, tp = confusion_matrix(truth.ravel(), (response >= .5).ravel(), labels=[False, True]).ravel().tolist()
                    assert (tp, fp, fn) == (scores['tp'], scores['fp'], scores['fn'])
                    ap = float(average_precision_score(truth.ravel(), response.ravel()))
                    internal_ap = binary_ranking_metrics(torch.from_numpy(response).flatten(), torch.from_numpy(truth).flatten())['average_precision']
                    assert abs(ap - internal_ap) < 2e-6
                    rows[name][arm][track] = dict(**scores, tn=tn, accuracy=(tp+tn)/truth.size, average_precision=ap,
                                                independent_confusion_exact=True, ap_implementation_difference=abs(ap-internal_ap))
            old = read(SYSTEM / 'nvos' / name / 'result.json')['metrics']
            assert abs(rows[name]['v10']['retained_gt_gt127']['iou'] - old['iou']) < 1e-7
            run.json(name + '_evaluation.json', rows[name])
            print('scored', name, flush=True)
        macro = {t: {a: {m: float(np.mean([r[a][t][m] for r in rows.values()])) for m in
                           ('iou', 'precision', 'recall', 'average_precision', 'accuracy')} for a in config['arms']} for t in config['gt_tracks']}
        other = {}
        for job in jobs:
            if job['benchmark'] != 'nvos':
                path = SYSTEM / job['benchmark'] / job['name'] / 'result.json'
                other[job['benchmark'] + '/' + job['name']] = dict(path=str(path), sha256=sha(path), intervention=False)
        run.json('summary.json', dict(complete=True, rows=rows, macro=macro, gt_hashes=gt_hashes,
            other_benchmarks=other, other_benchmarks_freshly_evaluated=False, default_changed=False,
            primary_arm=config['primary_arm'], target_rgb_arms=config['target_rgb_arms'],
            target_gt_used_for_prediction=False, independent_test=False, per_task_winners=False,
            prediction_seal_sha256=sha(run.root / 'predictions_complete.json'), durable_dense_prediction_bytes=0))
        print(json.dumps(macro, indent=2), flush=True)
    except Exception as error:
        run.json('failure.json', dict(type=type(error).__name__, message=str(error), completed_predictions=len(reports), target_gt_opened=access['gt']))
        raise
    finally:
        bank.close()


if __name__ == '__main__':
    main()
