"""Fixed-current-map single-level readout sensitivity; never an official row.

This deliberately does not fabricate three semantic scales or raw OpenCLIP
relevance from the current calibrated posterior. Old protocol/results stay intact.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

from radio_gs.utils.immutable_artifacts import sha256_file

BASE = Path('output/lerf_readout_sensitivity_next32')
PARENT = Path('output/full_field_rebuild_next19')
SCENES = ('figurines', 'ramen', 'teatime', 'waldo_kitchen')
COUNTS = dict(figurines=56, ramen=71, teatime=59, waldo_kitchen=22)
EVALUATOR = 'radio_gs/scripts/eval_prerendered_lerf_features.py'
FREEZE = 'paper/artifacts/evaluation_protocol_freeze_20260801.yaml'


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)


def register():
    BASE.mkdir(exist_ok=False)
    bindings = {p: sha256_file(p) for p in [__file__, EVALUATOR, FREEZE]}
    for scene in SCENES:
        root = PARENT / scene / 'evaluation'
        gp = Path('output/depth_opacity_repair_20260918') / scene / 'geometry_report.json'
        for path in [root / 'report.json', root / 'scores.pt', gp]:
            bindings[str(path)] = sha256_file(path)
        from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES
        for frame in SCENE_GT_FRAMES[scene]:
            path=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/scene/(frame+'.json')
            bindings[str(path)]=sha256_file(path)
    save(BASE / 'registration.json', dict(
        protocol_id='current_single_posterior_occam_operators_sensitivity_next32',
        official_protocol_equivalent=False, independent_test=False,
        scores='Unchanged final bounded_fallback_admission per-element calibrated posterior; exact original renderer/observed normalization and annotation grid.',
        arms=['unchanged_probability_gt_0.5', 'single_level_occam_30_7_operators'],
        comparator='Existing evaluate_relevance_maps: OpenCV30 activation, legacy7 mask smoothing, minmax activation and fixed0.5. One level only; no invented scales/OpenCLIP relevance.',
        authority='Diagnostic of output-contract effect only. No method promotion, threshold tuning, scene/query winners, target RGB, model training or modification of frozen rows.',
        scenes=SCENES, expected_observations=COUNTS, bindings=bindings))


def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = str(args[0])
        if any(x in path for x in ('teacher_radio.pt', 'teacher_dino.pt', 'teacher_siglip.pt', 'teacher_sam.pt', 'episodes.pt', 'caption_region_evidence.pt', 'grounded_text_supervision.pt')):
            raise RuntimeError('Training evidence forbidden during fixed-map readout')
        if '/lerf_ovs/label/' in path and path.lower().endswith(('.png', '.jpg')):
            raise RuntimeError('Target RGB forbidden')


def render_scene(scene, device, registration):
    from gsplat import rasterization
    from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
    from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
    from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects
    from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES

    dst = BASE / scene
    dst.mkdir(exist_ok=False)
    parent = PARENT / scene / 'evaluation'
    report = json.loads((parent / 'report.json').read_text())
    assert sha256_file(parent / 'scores.pt') == report['scores_sha256']
    pack = torch.load(parent / 'scores.pt', map_location='cpu', weights_only=False)
    assert pack['field_sha256'] == report['field_sha256']
    gp = Path('output/depth_opacity_repair_20260918') / scene / 'geometry_report.json'
    gr = json.loads(gp.read_text()); geometry_info = gr['results']['independent_scale']
    geometry = FrozenGaussianGeometry.load(geometry_info['geometry'], expected_sha256=geometry_info['geometry_sha256'], device=device)
    geometry.assert_binding(pack['geometry_binding'])
    prior = MogeGeometryPrior(gr['source_report'], gr['source_report_sha256'])
    observed = pack['observed'].to(device)
    scores = pack['scores']['final'].to(device)
    texts = pack['texts']
    frames = load_lerf_objects(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label') / scene, frames=SCENE_GT_FRAMES[scene])
    assert sum(map(len, frames.values())) == COUNTS[scene]
    outputs = {}
    with torch.no_grad():
        for frame, objects in frames.items():
            height, width = objects[0].mask.shape
            camera = prior.cameras[int(frame.split('_')[-1])]
            intrinsic = camera.intrinsic.to(device).clone()
            intrinsic[0] *= width / camera.width; intrinsic[1] *= height / camera.height
            kwargs = dict(viewmats=camera.world_to_camera.to(device)[None], Ks=intrinsic[None], width=width, height=height, near_plane=.001*prior.scale, far_plane=100*prior.scale, packed=False)
            values = []
            for obj in objects:
                colors = torch.stack([scores[:, texts.index(obj.query)], observed.float()], 1)
                render, alpha, _ = rasterization(means=geometry.xyz, quats=geometry.quaternions, scales=geometry.scales, opacities=geometry.opacities.flatten(), colors=colors, **kwargs)
                mass = render[0, ..., 1]
                valid = (mass / alpha[0, ..., 0].clamp_min(1e-8) > .5) & (alpha[0, ..., 0] > 1e-8)
                probability = torch.where(valid, render[0, ..., 0] / mass.clamp_min(1e-8), 0).cpu().numpy()
                # Test the actual emitted old prediction, not only its average metric.
                old_path=parent / 'final/lerf2d' / (frame + '_' + obj.query + '.png')
                old_row=next(r for r in report['lerf2d']['final']['observations'] if r['frame']==frame and r['query']==obj.query)
                assert sha256_file(old_path)==old_row['prediction_sha256']
                original = np.asarray(Image.open(old_path)) > 0
                assert np.array_equal(probability > .5, original), (scene, frame, obj.query, 'baseline prediction mismatch')
                values.append(probability)
            path = dst / (frame + '.npz')
            np.savez_compressed(path, probability=np.stack(values), queries=np.asarray([o.query for o in objects]))
            outputs[str(path)] = sha256_file(path)
            print(scene, frame, 'continuous predictions saved; baseline masks exact', flush=True)
    save(dst / 'render_receipt.json', dict(complete=True, observed_queries=COUNTS[scene], baseline_masks_bit_exact=True,
        parent_report_sha256=sha256_file(parent/'report.json'), geometry_sha256=geometry_info['geometry_sha256'],
        source_scores_sha256=sha256_file(parent/'scores.pt'), registration_sha256=sha256_file(BASE/'registration.json'), outputs=outputs))


def score_scene(scene):
    from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects, evaluate_relevance_maps, _mask_iou, _localization_hit
    from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES
    dst = BASE / scene
    receipt = json.loads((dst / 'render_receipt.json').read_text())
    for path, digest in receipt['outputs'].items():
        assert sha256_file(path) == digest
    frames = load_lerf_objects(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label') / scene, frames=SCENE_GT_FRAMES[scene])
    baseline, candidate = [], []
    for frame, objects in frames.items():
        with np.load(dst / (frame + '.npz'), allow_pickle=False) as bundle:
            assert list(bundle['queries']) == [o.query for o in objects]
            probability = bundle['probability']
        for obj, raw in zip(objects, probability):
            tensor = torch.from_numpy(raw)
            baseline.append(dict(frame=frame, query=obj.query, iou=_mask_iou(torch.from_numpy(obj.mask), tensor>.5), loc_hit=_localization_hit(torch.nonzero(tensor==tensor.max()), obj.bboxes)))
        result = evaluate_relevance_maps({frame: objects}, {frame: probability[None]}, mask_thresh=.5,
            activation_kernel=30, smooth_kernel=7, filter_implementation='opencv_filter2d',
            mask_smoothing_implementation='langsplat_legacy', resize_policy='error_on_mismatch')
        candidate.extend([dict(frame=frame, **row) for row in result['frames'][frame]['objects']])
    def summarize(rows):
        return dict(iou=float(np.mean([r['iou'] for r in rows])), localization=float(np.mean([r['loc_hit'] for r in rows])), observations=len(rows), rows=rows)
    original = json.loads((PARENT/scene/'evaluation/report.json').read_text())['lerf2d']['final']
    b = summarize(baseline)
    assert abs(b['iou']-original['miou']) < 1e-12
    assert abs(b['localization']-original['localization_accuracy']) < 1e-12
    save(dst/'report.json', dict(scene=scene, baseline=b, single_level_occam_operators=summarize(candidate),
        official_protocol_equivalent=False, independent_test=False, baseline_replay_exact=True,
        receipt_sha256=sha256_file(dst/'render_receipt.json'), registration_sha256=sha256_file(BASE/'registration.json')))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--register', action='store_true'); parser.add_argument('--scene', choices=SCENES); parser.add_argument('--device', default='cuda:0'); parser.add_argument('--summarize', action='store_true')
    args=parser.parse_args(); torch.set_num_threads(4)
    if args.register:
        return register()
    registration=json.loads((BASE/'registration.json').read_text())
    for path, digest in registration['bindings'].items():
        assert sha256_file(path)==digest, path
    if args.summarize:
        reports={s:json.loads((BASE/s/'report.json').read_text()) for s in SCENES}
        assert sum(r['baseline']['observations'] for r in reports.values())==208
        result=dict(complete=True, observations=208, official_protocol_equivalent=False, independent_test=False,
            scene_macro={arm:{key:float(np.mean([r[arm][key] for r in reports.values()])) for key in ('iou','localization')} for arm in ('baseline','single_level_occam_operators')},
            reports={str(BASE/s/'report.json'):sha256_file(BASE/s/'report.json') for s in SCENES}, registration_sha256=sha256_file(BASE/'registration.json'))
        save(BASE/'summary.json', result); print(json.dumps(result, indent=2)); return
    assert args.scene
    sys.addaudithook(guard)
    render_scene(args.scene, args.device, registration)
    score_scene(args.scene)


if __name__=='__main__':
    main()
