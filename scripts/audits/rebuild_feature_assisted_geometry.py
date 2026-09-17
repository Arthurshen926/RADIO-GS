"""Bounded independent geometry preprocessing: fixed feature veto vs same solver."""
import argparse
import json
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.geometry.fixed_feature_correspondence import fixed_feature_correspondences
from radio_gs.geometry.joint_scale import fit_joint_depth_scales
from radio_gs.scripts.repair_moge_geometry import load_views, build_geometry, residual_summary
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.audit_geometry_edge_transport import depth_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    path = base / 'geometry_only_repair_final_20260914' / a.scene / 'report.json'
    report = json.loads(path.read_text())
    prior = MogeGeometryPrior(report['source_report'], report['source_report_sha256'])
    all_records = prior.records
    feature_root = base / 'geometry_source_dino_20260917' / a.scene
    manifest = json.loads((feature_root / 'manifest.json').read_text())
    assert manifest['geometry_report_sha256'] == sha256_file(path)
    assert manifest['mapping_frames'] == report['mapping_frames']
    lookup = {r['frame']: r for r in manifest['records']}
    prior.records = {f: all_records[f] for f in report['mapping_frames']}
    views = load_views(prior, report['width'], a.device, allow_global_scale_fallback=True)
    recovered = {r['frame']: r for r in report['recovered_scales'] if r['accepted']}
    features = []
    for v in views:
        if not v['local_calibration_accepted']:
            v['depth'] *= recovered[v['frame']]['scale']
        r = lookup[v['frame']]
        assert r['image_sha256'] == prior.records[v['frame']]['source_image_sha256']
        assert sha256_file(r['path']) == r['sha256']
        features.append(torch.load(r['path'], map_location=a.device, weights_only=False))
    out = base / 'feature_assisted_geometry_20260917' / a.scene
    out.mkdir(parents=True, exist_ok=False)
    correspondences, audit = fixed_feature_correspondences(views, features, prior.scale)
    del features
    models = {}
    fits = {}
    for name, corr in correspondences.items():
        scales = fit_joint_depth_scales(*corr, num_views=len(views))
        fits[name] = dict(scales=scales.cpu().tolist(), pairs=len(corr[0]),
                          before=residual_summary(corr, torch.ones_like(scales)), after=residual_summary(corr, scales))
        model, _ = build_geometry(views, scales, prior.scale, report['voxel_metres'], a.device, report['minimum_voxel_views'])
        target = out / (name + '.pt')
        torch.save(dict(schema='radio_gs.frozen_surface_gaussian_geometry.v1',
            state_dict={k: v.cpu() for k,v in model.state_dict().items()}, geometry_trainable_during_semantics=False,
            mapping_frames=report['mapping_frames'], source_authority_sha256=prior.source_authority_sha256), target)
        fits[name].update(geometry_sha256=sha256_file(target), elements=len(model.xyz))
        models[name] = model
        print('built', name, fits[name], flush=True)
    del views, correspondences
    parent_path = base / 'affine_normal_geometry_20260917' / a.scene / 'geometry.pt'
    models['independent'] = FrozenGaussianGeometry.load(parent_path, expected_sha256=sha256_file(parent_path), device=a.device)
    prior.records = {f: all_records[f] for f in report['heldout_frames']}
    rows = []
    for v in load_views(prior, 480, a.device):
        h,w = v['depth'].shape
        row = dict(frame=v['frame'], variants={})
        for name, model in models.items():
            image,alpha,_ = rasterization(means=model.xyz, quats=model.quaternions, scales=model.scales,
                opacities=model.opacities.flatten(), colors=torch.zeros(len(model.xyz),1,device=a.device),
                viewmats=torch.linalg.inv(v['pose'])[None], Ks=v['k'][None], width=w,height=h,
                near_plane=.001*prior.scale, far_plane=100*prior.scale, render_mode='RGB+ED', packed=False)
            row['variants'][name] = depth_stats(image[0,...,-1],alpha[0,...,0],v,prior.scale)
        rows.append(row)
        print(v['frame'], {k:(r['coverage'],r['absrel']) for k,r in row['variants'].items()},flush=True)
    (out/'report.json').write_text(json.dumps(dict(schema='fixed_feature_geometry_control.v1', scene=a.scene,
        source_only=True, benchmark_data_used=False, semantic_field_used=False, heldout_feature_used=False,
        fitting_policy='same stride4, temporal-neighbor pairs, 2pixel reprojection cycle, depth/normal gates; optional fixed DINO cosine>=.6; same robust anchored +/-10percent scale solver',
        source_feature_manifest_sha256=sha256_file(feature_root/'manifest.json'), independent_geometry_sha256=sha256_file(parent_path),
        fits=fits, correspondence_audit=audit, rows=rows, promoted=False,
        implementation_sources={p:sha256_file(p) for p in [__file__, 'radio_gs/geometry/fixed_feature_correspondence.py', 'radio_gs/geometry/joint_scale.py', 'radio_gs/scripts/repair_moge_geometry.py']},
        limitation='MoGe references, not dense GT; feature similarity is a rejection cue only; all projection/semantic assets must be rebuilt if accepted.'),indent=2)+'\n')


if __name__ == '__main__':
    main()
