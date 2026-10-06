"""Source-only finer fusion with frozen calibration and heldout evaluation.

No semantic masks/queries/fields are consumed. Legacy-normal reconstruction
checks the parent, then matched corrected-normal 1cm and 5mm controls run.
"""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.scripts.repair_moge_geometry import load_views, build_geometry, evaluate_geometry
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, choices=['figurines', 'ramen'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path('output/finer_surface_carrier_20260918')/args.scene
    out.mkdir(parents=True, exist_ok=False)
    gp = Path('output/geometry_only_repair_final_20260914')/args.scene/'report.json'
    gr = json.loads(gp.read_text())
    prior = MogeGeometryPrior(gr['source_report'], gr['source_report_sha256'])
    old_spec = gr['results']['independent_scale']
    old = torch.load(old_spec['geometry'], weights_only=False)
    assert sha256_file(old_spec['geometry']) == old_spec['geometry_sha256']
    assert old['mapping_frames'] == gr['mapping_frames']
    assert bool((old['scales'] == 1).all()), 'This experiment binds independent-scale parent'
    recovery = {r['frame']:r for r in gr['recovered_scales']}
    result = dict(schema='finer_surface_carrier.v1', scene=args.scene, parent_report_sha256=sha256_file(gp), source_report=gr['source_report'], source_report_sha256=gr['source_report_sha256'], source_authority_sha256=gr['source_authority_sha256'], mapping_frames=gr['mapping_frames'], heldout_frames=gr['heldout_frames'], queries_opened=False, semantic_features_opened=False, semantic_masks_opened=False, geometry_trainable_during_semantics=False, results={}, implementation_sha256=sha256_file(__file__), geometry_builder_sha256=sha256_file('radio_gs/scripts/repair_moge_geometry.py'))
    for legacy in [True, False]:
        views = load_views(prior, gr['width'], 'cuda', True, legacy_affine_normals=legacy)
        by_frame = {v['frame']:v for v in views}
        training = [by_frame[f] for f in gr['mapping_frames']]
        evaluations = [by_frame[f] for f in gr['heldout_frames']]
        for v in training:
            if not v['local_calibration_accepted']:
                r = recovery[v['frame']]
                assert r['accepted']
                v['depth'] = v['depth']*r['scale']
        print(args.scene, 'loaded', 'legacy' if legacy else 'corrected', flush=True)
        for voxel in ([.01] if legacy else [.01, .005]):
            name = 'legacy_rebuild_10mm' if legacy else ('corrected_10mm' if voxel == .01 else 'corrected_5mm')
            model, _ = build_geometry(training, torch.ones(len(training), device='cuda'), prior.scale, voxel, 'cuda', gr['minimum_voxel_views'])
            path = out/(name+'.pt')
            torch.save(dict(schema='radio_gs.frozen_surface_gaussian_geometry.v1', state_dict={k:v.cpu() for k,v in model.state_dict().items()}, mapping_frames=gr['mapping_frames'], source_report_sha256=gr['source_report_sha256'], source_authority_sha256=gr['source_authority_sha256'], scales=torch.ones(len(training)), geometry_trainable_during_semantics=False), path)
            score = evaluate_geometry(model, evaluations, prior.scale)
            score.update(elements=len(model.xyz), geometry=str(path.resolve()), geometry_sha256=sha256_file(path), voxel_metres=voxel, normal_contract='legacy_no_affine_jacobian' if legacy else 'inverse_transpose_affine_jacobian')
            if legacy:
                same_shape = model.xyz.shape == old['state_dict']['xyz'].shape
                score['parent_reproduction'] = dict(same_shape=same_shape, xyz_max_error=float((model.xyz.cpu()-old['state_dict']['xyz']).abs().max()) if same_shape else None, aggregate_parent=gr['results']['independent_scale']['aggregate'])
            result['results'][name] = score
            (out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
            print(args.scene, name, score['aggregate'], score['elements'], flush=True)
            del model
            torch.cuda.empty_cache()
        del views, by_frame, training, evaluations
    result['complete'] = True
    result['limitations'] = 'Frozen source calibration and recovery reused, not independently re-estimated. Heldout frames excluded from fusion/recovery but old calibration authority is shared. Heldout MoGe is a proxy, not GT. New element identities require complete semantic asset rebuild; no old field reused.'
    (out/'report.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
