"""Matched native-score ranking diagnostic; no reused scope or probability heads."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects, _localization_hit
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    saved = False
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if '/lerf_ovs/label/' in path and (not saved or path.endswith(('.png','.jpg'))):
                raise RuntimeError('Target evidence before prediction or target RGB read')
            if any(x in path for x in ('teacher_siglip.pt','teacher_sam.pt','teacher_dino.pt','teacher_radio.pt','episodes.pt')):
                raise RuntimeError('Training evidence at query time')
    sys.addaudithook(guard)
    for scene in a.scenes:
        saved = False
        root = Path('output/native_residual_basis_next30_v2')/('lerf_'+scene)
        mr = json.loads((root/'map/report.json').read_text())
        assert sha256_file(root/'map/map.pt') == mr['map_sha256']
        data = torch.load(root/'map/map.pt', map_location='cuda', weights_only=False)
        fieldroot = Path(mr['root'])
        parent = fieldroot.parent.parent/'evaluation'
        receipt = json.loads((parent/'text_online.routes.json').read_text())
        qp = parent/'text_online.encodings.pt'
        assert sha256_file(qp) == receipt['encoding_bank_sha256']
        bank = torch.load(qp, map_location='cpu', weights_only=False)
        assert len(bank) == 1 and bank[0]['kind'] == 'text'
        q = F.normalize(bank[0]['encoding'].half().float().cuda(), dim=-1)
        texts = bank[0]['texts']
        scores = {name: (F.normalize(data['features'][:, :d], dim=-1) @ (q @ data['basis'][:d].T).T).cpu()
                  for name,d in [('native64',64),('native_residual',len(data['basis']))]}
        out = root/'ranking_evaluation'
        out.mkdir(exist_ok=False)
        torch.save(dict(scores=scores, texts=texts, map_sha256=mr['map_sha256'], query_sha256=sha256_file(qp)), out/'element_scores.pt')
        gp = Path('output/depth_opacity_repair_20260918')/scene/'geometry_report.json'
        gr = json.loads(gp.read_text())
        geo = FrozenGaussianGeometry.load(gr['results']['independent_scale']['geometry'],
            expected_sha256=gr['results']['independent_scale']['geometry_sha256'], device='cuda')
        geo.assert_binding(data['geometry_binding'])
        prior = MogeGeometryPrior(gr['source_report'], gr['source_report_sha256'])
        saved = True
        frames = load_lerf_objects(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/scene, frames=SCENE_GT_FRAMES[scene])
        assert set(texts) == {obj.query for objs in frames.values() for obj in objs}
        rows = {name: [] for name in scores}
        for frame, objects in frames.items():
            h,w = objects[0].mask.shape
            camera = prior.cameras[int(frame.split('_')[-1])]
            k = camera.intrinsic.cuda().clone()
            k[0] *= w/camera.width
            k[1] *= h/camera.height
            for name, values in scores.items():
                for obj in objects:
                    colors = torch.stack([values[:, texts.index(obj.query)].cuda()*data['observed'], data['observed'].float()], 1)
                    render, alpha, _ = rasterization(means=geo.xyz, quats=geo.quaternions, scales=geo.scales,
                        opacities=geo.opacities.flatten(), colors=colors, viewmats=camera.world_to_camera.cuda()[None],
                        Ks=k[None], width=w, height=h, near_plane=.001*prior.scale, far_plane=100*prior.scale, packed=False)
                    mass = render[0,...,1]
                    valid = (mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
                    response = torch.where(valid, render[0,...,0]/mass.clamp_min(1e-8), -1).cpu()
                    rows[name].append(dict(frame=frame, query=obj.query,
                        ap=float(average_precision_score(obj.mask.flatten(), response.numpy().flatten())),
                        localization_hit=_localization_hit(torch.nonzero(response==response.max()),obj.bboxes)))
            print(scene, frame, flush=True)
        report = dict(scene=scene, diagnostic_only=True, native_score_ranking=True, final_iou_not_measured=True,
            independent_test=False, target_masks_opened_after_element_scores=True, source_selected_before_benchmark=True,
            map_report_sha256=sha256_file(root/'map/report.json'), query_sha256=sha256_file(qp),
            scores_sha256=sha256_file(out/'element_scores.pt'), geometry_report_sha256=sha256_file(gp),
            rows=rows, metrics={n:dict(macro_ap=float(np.mean([r['ap'] for r in rs])),
                localization_accuracy=float(np.mean([r['localization_hit'] for r in rs])), observations=len(rs)) for n,rs in rows.items()},
            implementation_sha256=sha256_file(__file__))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(scene, report['metrics'], flush=True)


if __name__ == '__main__':
    main()
