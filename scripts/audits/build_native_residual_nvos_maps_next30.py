"""Build query-independent orthogonal map extensions on unchanged pixel evidence."""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--keys', nargs='+', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    source = Path('output/native_residual_basis_next30_v2')
    registration = json.loads((source/'registration.json').read_text())
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if '.labels.ply' in path or '/lerf_ovs/label/' in path:
                raise RuntimeError('Target evidence forbidden during map construction')
    sys.addaudithook(guard)
    for row in registration['cohort']:
        if row['key'] not in a.keys:
            continue
        assert row['benchmark'] == 'nvos'
        root = Path(row['root'])
        out = source/row['key']/'map'
        out.mkdir(exist_ok=False)
        def load(path, digest):
            assert sha256_file(path) == digest, path
            return torch.load(path, map_location='cpu', weights_only=False)
        selected = json.loads((out.parent/'report.json').read_text())
        assert selected['teacher_manifest_sha256'] == sha256_file(root/'teachers.json')
        basis = load(out.parent/'basis.pt', selected['selected_basis_sha256'])['basis'].cuda()
        payload = load(root/row['run']/'field.pt', selected['parent_field_sha256'])
        state = payload['state_dict']
        assert torch.equal(basis[:64].cpu(), state['native_decoder.weight'].T)
        info = json.loads((root/'teachers.json').read_text())
        tr = info['teacher_files']['siglip']
        desc = F.normalize(load(root/tr['path'], tr['sha256'])['descriptors'].float().cuda(), dim=-1) @ basis.T
        receipt = json.loads((root/'episodes.json').read_text())
        ep = load(root/'episodes.pt', receipt['sha256'])
        assert ep['teacher_manifest_sha256'] == sha256_file(root/'teachers.json')
        assert ep['geometry_binding'] == payload['geometry_binding']
        auth = json.loads((root/'responsibility.json').read_text())
        assert info['authority_sha256'] == sha256_file(root/'responsibility.json')
        n = len(state['latent'])
        numerator = torch.zeros(n, len(basis), device='cuda')
        mass = torch.zeros(n, device='cuda')
        for frame in info['train_frames']:
            vi = info['frames'].index(frame)
            ids = torch.where(ep['proposal_views'] == vi)[0]
            weights = ep['region_probability'][ids].float().cuda()
            numerator.addmm_(weights.T, desc[ids])
            mass += weights.sum(0)
        features = numerator / mass[:, None].clamp_min(1e-8)
        observed = state['observed'].cuda()
        assert (mass[observed] > 0).all() and torch.isfinite(features).all()
        error = float((features[observed, :64]-state['latent'][state['observed'], :64].cuda()).abs().max())
        assert error < 2e-5, ('Source fusion differs; cannot isolate residual', error)
        # Preserve original coordinates bit for bit, including unobserved rows.
        features[~observed, 64:] = 0
        features[:, :64] = state['latent'][:, :64].cuda()
        torch.save(dict(features=features.cpu(), basis=basis.cpu(), observed=state.get('image_observed', observed.cpu()).cpu(), residual_supported=observed.cpu(),
            geometry_binding=payload['geometry_binding'], parent_field_sha256=selected['parent_field_sha256']), out/'map.pt')
        report = dict(**row, source_only=True, original_coordinates_exact=True, base_reconstruction_max_error=error,
            original_heads_compatible=False, measured_extension_residual_is_zero=True, fusion='conditional_region_probability', geometry_binding=payload['geometry_binding'], source_frames=info['train_frames'],
            basis_report_sha256=sha256_file(out.parent/'report.json'), map_sha256=sha256_file(out/'map.pt'),
            implementation_sha256=sha256_file(__file__), fusion_implementation_sha256=sha256_file('radio_gs/training/source_pixel_evidence.py'))
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(row['key'], 'map ready', list(features.shape), 'base error', error, flush=True)


if __name__ == '__main__':
    main()
