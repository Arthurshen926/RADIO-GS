"""Frozen field readout at the LERF2D annotation grid, using frozen metrics.

Preserves the method's posterior/.5 decision rule. Comparator-specific CLIP
normalization and smoothing are not transplanted into a calibrated posterior.
No target RGB, source teacher tensors or threshold fitting are allowed.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects, _mask_iou, _localization_hit
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.utils.query_input_identity import query_call_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', choices=list(SCENE_GT_FRAMES), required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if any(s in path for s in ('teacher_radio.pt', 'teacher_dino.pt', 'teacher_siglip.pt',
                'teacher_sam.pt', 'episodes.pt', 'caption_region_evidence.pt', 'grounded_text_supervision.pt')):
                raise RuntimeError('Training asset access forbidden')
            if '/lerf_ovs/label/' in path and path.lower().endswith(('.jpg', '.png')):
                raise RuntimeError('Target RGB forbidden')
    sys.addaudithook(guard)
    torch.set_num_threads(4)
    root = Path('output/depth_opacity_pipeline_20260918')/a.scene/'reconstruction/field'
    gp = Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json'
    run = 'bounded_fallback_admission'
    prior_report = json.loads(gp.read_text())
    g = prior_report['results']['independent_scale']
    geo = FrozenGaussianGeometry.load(g['geometry'], expected_sha256=g['geometry_sha256'], device=a.device)
    prior = MogeGeometryPrior(prior_report['source_report'], prior_report['source_report_sha256'])
    identity = json.loads((root/'native_query_encoder.json').read_text())
    authority = json.loads((root/'responsibility.json').read_text())
    assert authority['metadata']['frozen_geometry_report_sha256'] == sha256_file(gp)
    digest = json.loads((root/run/'report.json').read_text())['field_sha256']
    field, _ = load_element_retrieval_field(root/run/'field.pt', expected_sha256=digest,
        geometry=geo, expected_query_encoder=identity, device=a.device)
    base = Path('output/bounded_fallback_repair_20260918')/a.scene
    bankpath = base/'text_online.encodings.pt'
    receipt = json.loads((base/'text_online.routes.json').read_text())
    assert receipt['encoding_bank_sha256'] == sha256_file(bankpath)
    bank = torch.load(bankpath, map_location='cpu', weights_only=False)
    assert len(bank) == 1 and bank[0]['kind'] == 'text'
    texts = bank[0]['texts']
    assert bank[0]['query_input_sha256'] == query_call_sha256('text', (texts,), {}, identity)
    labels = Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/a.scene
    frames = load_lerf_objects(labels, frames=SCENE_GT_FRAMES[a.scene])
    assert set(frames) == set(SCENE_GT_FRAMES[a.scene]) and all(frames.values())
    assert set(texts) == {o.query for objects in frames.values() for o in objects}
    info = json.loads((root/'teachers.json').read_text())
    assert {int(f.split('_')[-1]) for f in frames}.isdisjoint(info['frames'])
    a.output.mkdir(parents=True)
    observations = []
    with torch.no_grad():
        response = field.retrieve(bank[0]['encoding'].half().float().to(a.device), 'text')['scores']
        for frame, objects in frames.items():
            h, w = objects[0].mask.shape
            camera = prior.cameras[int(frame.split('_')[-1])]
            k = camera.intrinsic.to(a.device).clone()
            k[0] *= w/camera.width
            k[1] *= h/camera.height
            for start in range(0, len(objects), 4):
                batch = objects[start:start+4]
                colors = torch.cat([response[:, [texts.index(o.query) for o in batch]], field.observed.float()[:, None]], 1)
                image, alpha, _ = rasterization(means=geo.xyz, quats=geo.quaternions,
                    scales=geo.scales, opacities=geo.opacities.flatten(), colors=colors,
                    viewmats=camera.world_to_camera.to(a.device)[None], Ks=k[None], width=w, height=h,
                    near_plane=.001*prior.scale, far_plane=100*prior.scale, packed=False)
                mass = image[0, ..., -1]
                valid = (mass/alpha[0, ..., 0].clamp_min(1e-8) > .5) & (alpha[0, ..., 0] > 1e-8)
                for j, obj in enumerate(batch):
                    score = torch.where(valid, image[0, ..., j]/mass.clamp_min(1e-8), 0).cpu()
                    binary = score > .5
                    path = a.output/f'{frame}_{obj.query}.png'
                    Image.fromarray(binary.numpy().astype(np.uint8)*255).save(path)
                    hit = _localization_hit(torch.nonzero(score == score.max()), obj.bboxes)
                    observations.append(dict(frame=frame, query=obj.query,
                        iou=_mask_iou(torch.from_numpy(obj.mask), binary), localization_hit=hit,
                        prediction_sha256=sha256_file(path), grid_hw=[h, w]))
            print(a.scene, frame, flush=True)
    geo.assert_binding(authority['metadata']['frozen_geometry_binding'])
    result = dict(contract='lerf2d-field-only-four-scene-v1', scene=a.scene,
        method_id='native96-depth-opacity-bounded-fallback-20260918', independent_test=False,
        field_sha256=digest, geometry_sha256=g['geometry_sha256'], threshold=.5,
        readout='Frozen absolute membership, full-scene observed-alpha normalization; native annotation grid.',
        localization='Frozen evaluator: any global maximum inside any annotation bounding box.',
        target_rgb_opened=False, threshold_search=False,
        metric_implementation_sha256=sha256_file('radio_gs/scripts/eval_prerendered_lerf_features.py'),
        implementation_sha256=sha256_file(__file__), encoding_bank_sha256=sha256_file(bankpath),
        labels={str(labels/(f+'.json')):sha256_file(labels/(f+'.json')) for f in frames},
        observation_count=len(observations), observations=observations,
        miou=sum(r['iou'] for r in observations)/len(observations),
        localization_accuracy=sum(r['localization_hit'] for r in observations)/len(observations))
    (a.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(a.scene, result['miou'], result['localization_accuracy'], flush=True)


if __name__ == '__main__':
    main()
