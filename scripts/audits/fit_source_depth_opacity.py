"""Bounded geometry-only opacity calibration; no semantics or queries."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.geometry.depth_opacity import depth_distribution_loss
from radio_gs.scripts.repair_moge_geometry import load_views, evaluate_geometry
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scene', required=True); args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path('output/depth_opacity_repair_20260918')/args.scene
    out.mkdir(parents=True, exist_ok=False)
    gp = Path('output/geometry_only_repair_final_20260914')/args.scene/'report.json'; gr = json.loads(gp.read_text())
    prior = MogeGeometryPrior(gr['source_report'], gr['source_report_sha256'])
    spec = gr['results']['independent_scale']
    model = FrozenGaussianGeometry.load(spec['geometry'], expected_sha256=spec['geometry_sha256'], device='cuda')
    original = {k:v.clone() for k,v in model.state_dict().items()}
    payload = torch.load(spec['geometry'], weights_only=False)
    assert payload['mapping_frames'] == gr['mapping_frames'] and bool((payload['scales'] == 1).all())
    views = load_views(prior, gr['width'], 'cuda', True, legacy_affine_normals=True)
    views = {v['frame']:v for v in views}; recovery = {r['frame']:r for r in gr['recovered_scales']}
    heldout = [views[f] for f in gr['heldout_frames']]
    training = [views[f] for f in gr['mapping_frames']]
    assert not set(gr['mapping_frames']) & set(gr['heldout_frames'])
    for v in training:
        if not v['local_calibration_accepted']:
            rec = recovery[v['frame']]; assert rec['accepted']; v['depth'] = v['depth']*rec['scale']
    before = evaluate_geometry(model, heldout, prior.scale)
    # Fixed 256-wide source render; two deterministic passes, no target selection.
    logits = torch.full((len(model.xyz),), torch.logit(torch.tensor(.8)).item(), device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=.1)
    history = []
    for step in range(2*len(training)):
        v = training[step % len(training)]; width = 256; height = round(v['depth'].shape[0]*width/v['depth'].shape[1])
        k = v['k'].clone(); k[0] *= width/v['depth'].shape[1]; k[1] *= height/v['depth'].shape[0]
        target = F.interpolate(v['depth'][None, None], size=(height, width), mode='nearest-exact')[0, 0]/prior.scale
        valid = F.interpolate(v['valid'][None, None].float(), size=(height, width), mode='nearest-exact')[0, 0].bool()
        vm = torch.linalg.inv(v['pose'])
        z = (model.xyz@vm[:3, :3].T+vm[:3, 3])[:, 2]/prior.scale
        colors = torch.stack([z, z.square()], 1)
        optimizer.zero_grad()
        render, alpha, _ = rasterization(means=model.xyz, quats=model.quaternions, scales=model.scales, opacities=logits.sigmoid(), colors=colors, viewmats=vm[None], Ks=k[None], width=width, height=height, near_plane=.001*prior.scale, far_plane=100*prior.scale, packed=False)
        loss, parts = depth_distribution_loss(render[0], alpha[0, ..., 0], target, valid)
        loss = loss+.001*(logits.sigmoid()-.8).square().mean()
        assert bool(torch.isfinite(loss))
        loss.backward(); assert bool(torch.isfinite(logits.grad).all()); optimizer.step()
        with torch.no_grad(): logits.clamp_(torch.logit(torch.tensor(.02)).item(), torch.logit(torch.tensor(.98)).item())
        if step % 20 == 0:
            row = dict(step=step, frame=v['frame'], loss=float(loss), **{k:float(x) for k,x in parts.items()}); history.append(row); print(args.scene, row, flush=True)
    model.opacities.copy_(logits.detach().sigmoid()[:, None])
    for key in original:
        if key != 'opacities': assert torch.equal(original[key], model.state_dict()[key])
    after = evaluate_geometry(model, heldout, prior.scale)
    path = out/'geometry.pt'; payload['state_dict'] = {k:v.cpu() for k,v in model.state_dict().items()}; torch.save(payload, path)
    report = dict(schema='source_depth_opacity.v1', scene=args.scene, parent_report_sha256=sha256_file(gp), parent_geometry_sha256=spec['geometry_sha256'], geometry=str(path.resolve()), geometry_sha256=sha256_file(path), implementation_sha256=sha256_file(__file__), loss_implementation_sha256=sha256_file('radio_gs/geometry/depth_opacity.py'), mapping_frames=gr['mapping_frames'], heldout_frames=gr['heldout_frames'], semantic_features_opened=False, queries_opened=False, semantic_masks_opened=False, all_non_opacity_tensors_exact=True, steps=2*len(training), learning_rate=.1, opacity_bounds=[.02,.98], before=before, after=after, training_log=history, limitations='Only source geometry depth and cameras used. Old calibration/recovery shared, heldout source MoGe is not independent GT. Opacity optimized only in isolated preprocessing; semantic reconstruction must freeze new geometry and rebuild all dependent assets. Fixed budget, no selection on reviewed masks or benchmark.')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(args.scene, 'before', before['aggregate'], 'after', after['aggregate'], flush=True)


if __name__ == '__main__':
    main()
