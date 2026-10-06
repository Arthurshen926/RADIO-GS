"""Complete frozen field recipe on one provided-mesh ScanNet source cohort.

Dataset adapters change cameras/geometry inputs only. No semantic evaluation is
performed here. Every stage is recorded; incomplete stages fail closed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from radio_gs.geometry.registered_pinhole import load_registered_camera_table
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    root = Path('output/scannet_frozen_field_20260918')/a.scene
    root.mkdir(parents=True, exist_ok=a.resume)
    marker = root/'execution.json'
    record = json.loads(marker.read_text()) if marker.exists() else dict(scene=a.scene,
        parent_method='native96-depth-opacity-bounded-fallback-20260918', input_track='provided_unlabelled_mesh',
        same_geometry_input_as_lerf=False, complete=False, stages=[])
    def run(name, command, outputs):
        command = list(map(str, command))
        old = next((r for r in record['stages'] if r['name'] == name), None)
        if old:
            assert old['command'] == command and old['status'] == 'complete'
            for path, digest in old['outputs'].items():
                assert sha256_file(path) == digest
            return
        row = dict(name=name, command=command, status='running')
        record['stages'].append(row)
        marker.write_text(json.dumps(record, indent=2)+'\n')
        print(a.scene, name, flush=True)
        with (root/(name+'.log')).open('x') as log:
            ret = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='4'))
        row.update(returncode=ret.returncode, status='failed' if ret.returncode else 'complete')
        if not ret.returncode:
            row['outputs'] = {str(f):sha256_file(f) for f in outputs}
        marker.write_text(json.dumps(record, indent=2)+'\n')
        if ret.returncode:
            raise RuntimeError(name+' failed; inspect preserved assets')
    base = root/'base'
    gp = root/'geometry_report.json'
    if not gp.exists():
        base.mkdir()
        meshreport = Path('output/scannet_provided_mesh_track_20260918')/a.scene/'geometry_report.json'
        g = json.loads(meshreport.read_text())
        assert sha256_file(g['geometry']) == g['geometry_sha256']
        cp = Path(g['source_cameras'])
        assert sha256_file(cp) == g['source_cameras_sha256']
        cameras = load_registered_camera_table(cp)
        ids = sorted(cameras)
        validation = [ids[len(ids)//3], ids[2*len(ids)//3]]
        eligible = [f for f in ids if f not in validation]
        poses = torch.stack([cameras[f].world_to_camera for f in eligible])
        centers = torch.stack([cameras[f].camera_to_world[:3, 3] for f in eligible])
        dist = torch.cdist(centers, centers).square()/torch.cdist(centers, centers).max().clamp_min(1e-8).square()
        dist += torch.cdist(poses[:, 2, :3], poses[:, 2, :3]).square()
        chosen = [0]
        while len(chosen) < 16:
            nearest = dist[:, chosen].amin(1)
            nearest[chosen] = -1
            chosen.append(int(nearest.argmax()))
        train = sorted(eligible[i] for i in chosen)
        frames = train+validation
        by = {r['frame_index']:r for r in json.loads(cp.read_text())['records']}
        images = []
        rgb = root/'source_rgb'
        rgb.mkdir()
        for f in frames:
            row = by[f]
            path = Path(row['source_image'])
            assert sha256_file(path) == row['source_sha256']
            shutil.copy2(path, rgb/path.name)
            images.append(dict(image_id=f'frame_{f:05d}', path=str(path), sha256=row['source_sha256'], rgb_role='registered_source_or_mapping_view'))
        authority = root/'sam_rgb_authority.json'
        authority.write_text(json.dumps(dict(schema_version=1, contract='sam3-query-free-source-rgb-authority-v1',
            information_policy=dict(registered_source_rgb_only=True, query_text_used=False,
                benchmark_ground_truth_used=False, target_or_evaluation_rgb_used=False), images=images), indent=2)+'\n')
        plan = dict(scene=a.scene, frames=frames, train_frames=train, validation_frames=validation,
            radio_root=str((root/'radio').resolve()), benchmark_data_used=False,
            source_authority=dict(path=str(authority.resolve()), sha256=sha256_file(authority)),
            selection='Fixed sorted-third validation, greedy source camera coverage train16; no labels.')
        (base/'source_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        report = dict(schema='registered_mesh_field_geometry.v1', input_track=g['input_track'],
            source_report=str(cp.resolve()), source_report_sha256=sha256_file(cp), mapping_frames=frames,
            results={'independent_scale':dict(geometry=g['geometry'], geometry_sha256=g['geometry_sha256'], elements=g['elements'])},
            mesh_report=str(meshreport), mesh_report_sha256=sha256_file(meshreport))
        gp.write_text(json.dumps(report, indent=2)+'\n')
    if a.prepare_only:
        marker.write_text(json.dumps(record, indent=2)+'\n')
        return
    plan = json.loads((base/'source_plan.json').read_text())
    py = sys.executable
    sam = '/root/miniconda3/envs/sam3/bin/python'
    guard = 'scripts/audits/run_scannet_source_stage.py'
    run('radio', [py, guard, '--module', 'radio_gs.scripts.extract_radio_features', '--scene', a.scene,
        '--image_dir', root/'source_rgb', '--output_dir', root/'radio', '--radio_checkpoint',
        '/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar', '--batch_size', '1', '--skip_pca_stats', '--resume-partial', '--device', a.device], [root/'radio/frame_manifest.json'])
    command = [sam, guard, '--module', 'radio_gs.scripts.build_sam3_multiscale_hierarchy_cache',
        '--source-authority', plan['source_authority']['path'], '--source-authority-sha256', plan['source_authority']['sha256'],
        '--image-ids', ','.join(f'frame_{f:05d}' for f in plan['frames']), '--output-root', base/'source_sam',
        '--checkpoint-path', 'checkpoints/sam3_modelscope/sam3.pt', '--checkpoint-sha256',
        '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e', '--device', a.device]
    run('sam', command, [base/'source_sam/manifest.json'])
    sr = base/'source_sam_execution.json'
    if not sr.exists():
        snap = base/'sam_sources'
        snap.mkdir()
        impl = {}
        for name in ['radio_gs/scripts/build_sam3_multiscale_hierarchy_cache.py', 'radio_gs/scripts/build_sam3_foundation_cache.py', 'radio_gs/models/sam3_multiscale_hierarchy.py', guard]:
            shutil.copy2(name, snap/Path(name).name)
            impl[name] = sha256_file(name)
        sr.write_text(json.dumps(dict(status='complete', command=list(map(str, command)), implementations=impl), indent=2)+'\n')
    run('base', [py, 'scripts/audits/run_registered_mesh_base.py', '--scene', a.scene, '--device', a.device,
        '--output-root', base, '--geometry-report', gp, '--until-base'], [base/'field/scale_affinity_v1_direct_control/report.json'])
    field = root/'reconstruction/field'
    run('rebuild', [py, 'scripts/audits/run_registered_mesh_pixel_rebuild.py', '--scene', a.scene, '--device', a.device,
        '--parent', base/'field', '--output-root', root/'reconstruction', '--geometry-report', gp, '--skip-evaluation'], [field/'joint_supervised_image_residual/report.json'])
    run('bounded_scope', [py, guard, 'scripts/audits/fit_bounded_fallback_scope.py', '--root', field,
        '--device', a.device, '--withheld-concepts'], [field/'bounded_fallback_withheld/report.json'])
    record['source_acceptance'] = json.loads((field/'bounded_fallback_withheld/report.json').read_text())['source_acceptance']
    record['evaluation_only_not_promotion'] = True
    record['rejected_candidates_remain_in_full_cohort'] = True
    run('admission', [py, guard, 'scripts/audits/fit_counterfactual_extent_admission.py', '--root', field,
        '--scene', a.scene, '--device', a.device, '--parent', 'bounded_fallback_withheld',
        '--name', 'bounded_fallback_admission', '--withheld-fallback'], [field/'bounded_fallback_admission/report.json'])
    record['complete'] = True
    marker.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
