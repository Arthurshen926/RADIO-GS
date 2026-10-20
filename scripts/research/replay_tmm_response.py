"""Replay all four retained LERF controls without saving duplicate readout tensors.

This verifies the extracted compatibility core, NOT accuracy of refine_response.
All original source frames are used. No ground truth is read or scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.response_refinement import fuse_admitted_evidence
from scripts.research.fit_consolidated_response import iter_source_response_views

SCENES = ('figurines', 'ramen', 'teatime', 'waldo_kitchen')
STUDY = Path('/mnt/pool/sqy/results/RADIO-GS/output/tmm_response_20261016/study')
POLICY = dict(study_bytes=2**31, run_bytes=2**28, artifact_bytes=2**24, reserve_bytes=2**30)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def replay(scene_name, run):
    started = time.monotonic()
    fit_path = Path('output/consolidated_response_control') / scene_name / 'fusion/fit.json'
    fit = json.loads(fit_path.read_text())
    inputs = json.loads(Path(fit['inputs']).read_text())
    scene = json.loads(Path(inputs['scene']).read_text())
    if scene['source_frames'] != fit['source_frames'] or fit['raster'] != [192, 256]:
        raise ValueError('retained observation contract changed')
    data = torch.load(fit['readout'], map_location='cpu', weights_only=False)
    binding = data['semantic_geometry']
    geo = FrozenGaussianGeometry.load(binding['path'], expected_sha256=binding['sha256'], device='cuda')
    gp = MogeGeometryPrior(scene['camera_report'], scene['camera_report_sha256'])
    observed, prior = data['semantic_observed'].cuda(), data['scores'].cuda()
    old_mass, old_numerator = torch.zeros_like(prior), torch.zeros_like(prior)
    active = torch.zeros(prior.shape[1], dtype=torch.long)
    count = 0

    def observations():
        nonlocal count
        for op, target, positive, negative, native_positive, confidence in iter_source_response_views(
                inputs, scene, data, geo, gp, observed, prior):
            known = (positive | (negative & ~native_positive)).float() * confidence[None]
            # Independent retained formula, same freshly constructed observations.
            old_mass.add_(torch.sparse.mm(op.T, known))
            old_numerator.add_(torch.sparse.mm(op.T, known * target))
            active.add_((confidence > 0).cpu().long())
            count += 1
            if count % 50 == 0:
                print(scene_name, count, '/', len(scene['source_frames']), flush=True)
            yield op, target, known

    torch.cuda.reset_peak_memory_stats()
    values, mass = fuse_admitted_evidence(prior, observations())
    old = torch.where(old_mass > 1e-8, old_numerator / old_mass.clamp_min(1e-8), prior)
    fresh_delta = float((values - old).abs().max())
    fresh_mass_delta = float((mass - old_mass).abs().max())
    # CUDA sparse transpose products use parallel accumulation. Mass grows with
    # view count; record absolute error, but check both relative and absolute
    # tolerances instead of treating the raw count as a bounded probability.
    fresh_mass_close = torch.allclose(mass, old_mass, rtol=1e-6, atol=1e-6)
    fresh_passed = fresh_delta <= 1e-6 and fresh_mass_close
    print(scene_name, 'fresh response/mass max error', fresh_delta, fresh_mass_delta, flush=True)
    if active.tolist() != fit['active_views_per_query']:
        raise AssertionError('active observation identities differ')
    values[~observed] = 0
    expected_path = fit_path.parent / 'native_protected/typed_readout.pt'
    expected = torch.load(expected_path, map_location='cpu', weights_only=False)
    actual = values.cpu()
    delta = (actual - expected['scores']).abs()
    selection_exact = all(torch.equal(data[key], expected[key]) for key in
                          ('selection_scores', 'selection_observed', 'semantic_observed'))
    if not selection_exact or data['texts'] != expected['texts']:
        raise AssertionError('retained identity or selected set differs')
    maximum = float(delta.max())
    passed = fresh_passed and maximum <= 2e-5
    receipt = dict(scene=scene_name, source_views=count, queries=len(data['texts']),
        fresh_formula_max_abs=fresh_delta, fresh_mass_max_abs=fresh_mass_delta,
        fresh_response_atol=1e-6, fresh_mass_atol=1e-6, fresh_mass_rtol=1e-6,
        fresh_mass_close=fresh_mass_close,
        fresh_formula_bit_equal=torch.equal(values[observed], old[observed]),
        retained_response_max_abs=maximum, retained_response_mean_abs=float(delta.mean()),
        retained_response_bit_equal=torch.equal(actual, expected['scores']),
        retained_half_threshold_crossings=int(((actual > .5) != (expected['scores'] > .5)).sum()),
        retained_tolerance=2e-5, passed=passed, selection_exact=selection_exact,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic()-started,
        source_geometry=binding, source_files={str(p): sha(p) for p in
            (fit_path, Path(fit['inputs']), Path(inputs['scene']), Path(fit['readout']), expected_path)},
        new_teacher_calls=0, target_gt_read=False, tensors_written=0,
        new_directional_refinement_accuracy_tested=False)
    run.json(scene_name+'.json', receipt)
    print(json.dumps(receipt), flush=True)
    if not passed:
        raise AssertionError('retained response tolerance exceeded; inspect report')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='legacy_core_replay')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = BoundedExperiment(STUDY, args.run, POLICY)
    source_files = [Path(__file__), Path('scripts/research/fit_consolidated_response.py'),
                    Path('radio_gs/querying/response_refinement.py')]
    run.json('manifest.json', dict(scenes=SCENES, source_files={str(p): sha(p) for p in source_files},
        torch_version=torch.__version__, device=torch.cuda.get_device_name(),
        purpose='full-source numerical extraction replay, not a new accuracy experiment'))
    rows = []
    try:
        for scene in SCENES:
            rows.append(replay(scene, run))
            torch.cuda.empty_cache()
        run.json('summary.json', dict(complete=True, rows=rows, all_passed=all(r['passed'] for r in rows),
            storage_before_summary_bytes=tree_bytes(STUDY), duplicate_tensor_bytes=0))
    except Exception as error:
        run.json('failure.json', dict(completed=[r['scene'] for r in rows], error=repr(error)))
        raise


if __name__ == '__main__':
    main()
