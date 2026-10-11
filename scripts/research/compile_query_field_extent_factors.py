"""Compile complete construction phases, with no benchmark query reads."""
import argparse
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from radio_gs.query_field.storage import load_field, save_field, sha256
from radio_gs.query_field.extent_factors import ExtentFactors, ExtentView, compile_extent_view
from radio_gs.query_field.coverage_factors import CoverageView, compile_coverage_view
from radio_gs.query_field.geometry import relaxed_subset_alpha
from scripts.research.compile_query_field_coverage_factors import member_conditions
from scripts.research.run_query_field_relations import read, write, receipt

ROOT = Path('output/query_field_20261011/compiled_extent_factors_v2')
CONFIG = Path('output/query_field_20261011/binary_region_scope_v2/prediction_config.json')
PROTOCOL = Path('docs/experiments/2026-10-11-extent-decision-v2-protocol.md')


def phase_pixels(vi):
    return torch.arange(384*512).reshape(384, 512)[(vi%4)//2::2, (vi%4)%2::2].flatten()


def run(index):
    start = time.monotonic(); j = read(CONFIG)['jobs'][index]
    out = ROOT/j['scene']; out.mkdir(parents=True, exist_ok=False)
    print('loading', j['scene'], flush=True)
    field, parent = load_field(j['field'])
    domain = field.observed | field.semantic_measure.supported
    geo = SimpleNamespace(**{k: getattr(field, k).cuda() for k in ['xyz', 'quaternions', 'scales', 'opacities']},
                          observed=domain.cuda(), world_units_per_metre=field.world_units_per_metre)
    construction = read(j['construction']); observations = read(Path(j['observations'])/'observations.json')
    frames = [r['frame'] for r in construction['source_records']]
    assert len(frames) == 18 and set(frames) <= set(observations['splits']['train_frames'])
    assert not set(frames) & set(observations['splits']['development_frames'])
    state = {}; checks = []
    for vi, frame in enumerate(frames):
        path = Path(j['observations'])/f'view_{frame:05d}.pt'
        entry = next(r for r in observations['frames'] if r['frame'] == frame)
        digest = sha256(path); assert digest == entry['compiled']['sha256']
        raw = torch.load(path, weights_only=True)
        h, w = raw['original_hw']
        camera = SimpleNamespace(intrinsic=raw['intrinsic'], world_to_camera=raw['world_to_camera'], height=h, width=w)
        del raw
        pixels = phase_pixels(vi)
        compiled = compile_extent_view(geo, camera, 384, 512, pixels, member_domain=geo.observed)
        # Independent old compiler on a fixed query-independent 64-ray lattice.
        sample = torch.linspace(0, len(pixels)-1, 64).long()
        small = compile_coverage_view(geo, camera, 384, 512, pixels[sample], torch.zeros(64, dtype=torch.long), member_domain=geo.observed)
        for si, r in enumerate(sample):
            a, b = compiled['pointer'][r:r+2]; c, d = small['pointer'][si:si+2]
            assert torch.equal(compiled['index'][a:b], small['index'][c:d]), (frame, 'index')
            assert torch.allclose(compiled['alpha'][a:b], small['alpha'][c:d], atol=1e-5, rtol=0), (frame, 'alpha')
        operator = ExtentView(compiled, len(geo.xyz), geo.opacities).cuda()
        depth = geo.xyz @ camera.world_to_camera[2, :3].cuda() + camera.world_to_camera[2, 3].cuda()
        cases = []
        for name, member in member_conditions(geo, vi, depth > depth.median()):
            with torch.no_grad():
                predicted = operator(member)
                reference = relaxed_subset_alpha(geo, member, camera, 384, 512, geo.world_units_per_metre, member_domain=geo.observed)[pixels.cuda()]
            error = float((predicted-reference).abs().max())
            different = int(((predicted > 10.5/255) != (reference > 10.5/255)).sum())
            assert error <= 1e-4 and different == 0, (j['scene'], vi, name, error, different)
            cases.append(dict(condition=name, max_alpha_error=error, selection_disagreements=different))
            if name == 'random50':
                from radio_gs.query_field.extent_decision import render_binary
                cpu = torch.from_numpy(render_binary(compiled['pointer'].numpy(), compiled['index'].numpy(), compiled['alpha'].numpy(), (member.cpu().numpy() > .5)))
                cpu_error = float((cpu-predicted.cpu()).abs().max())
                assert cpu_error <= 1e-6 and torch.equal(cpu > 10.5/255, predicted.cpu() > 10.5/255), (vi, cpu_error)
                cases[-1]['cpu_decision_alpha_error'] = cpu_error
        state.update({f'views.{vi}.'+k: v for k, v in compiled.items()})
        checks.append(dict(frame=frame, source_sha256=digest, rays=len(pixels), hits=len(compiled['index']), cases=cases, old_compiler_index_exact=64))
        print('compiled', j['scene'], vi, 'hits', len(compiled['index']), 'max_error', max(c['max_alpha_error'] for c in cases), flush=True)
        del operator, predicted, reference, member
    field.extent_factors = ExtentFactors(state, len(field.xyz), field.opacities)
    added = sum(v.numel()*v.element_size() for v in state.values())
    exported = save_field(field, out/'field', dict(parent_field_sha256=parent['files']['field.pt']['sha256'],
        benchmark_queries_used=False, target_labels_used=False, cameras_in_export=False, default_decoder_changed=False,
        source_records=checks, scope_targets_in_export=False, complete_construction_phase=True))
    assert exported['tensor_bytes'] == parent['tensor_bytes'] + added
    write(out/'compilation.json', dict(complete=True, scene=j['scene'], parent_field=receipt(Path(j['field'])/'manifest.json'),
        field_sha256=exported['files']['field.pt']['sha256'], views=len(checks), rays=sum(c['rays'] for c in checks),
        hits=sum(c['hits'] for c in checks), added_bytes=added, checks=checks, seconds=time.monotonic()-start,
        protocol=receipt(PROTOCOL), implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        queries_used=False, scope_targets_in_export=False, standard_decoder_changed=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--index', type=int, required=True); a = p.parse_args()
    torch.set_num_threads(4); run(a.index)
