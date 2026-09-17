"""Source-only controls for proposal-count and grounding-inventory bias.

Same semantic basis, structural field, support and calibrators as the parent.
No benchmark labels or queries are used to build these analytical controls.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--origin-balanced', action='store_true')
    parser.add_argument('--modes', nargs='+', choices=['queryfree','view_balanced','view_origin_balanced','origin_balanced'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.root
    parent = root / 'scale_affinity_v1_direct_control'
    info = json.loads((root / 'teachers.json').read_text())
    receipt = json.loads((root / 'episodes.json').read_text())
    assert sha256_file(root / 'episodes.pt') == receipt['sha256']
    data = torch.load(root / 'episodes.pt', weights_only=False)
    assert sha256_file(root / 'teachers.json') == data['teacher_manifest_sha256']
    report = json.loads((parent / 'report.json').read_text())
    assert sha256_file(parent / 'field.pt') == report['field_sha256']
    payload = torch.load(parent / 'field.pt', weights_only=False)
    assert payload['geometry_binding'] == data['geometry_binding']
    loaded = {}
    for name in ['sam', 'siglip']:
        record = info['teacher_files'][name]
        assert sha256_file(root / record['path']) == record['sha256']
        loaded[name] = torch.load(root / record['path'], weights_only=False)
    original_count = loaded['sam']['source_receipts'][0]['all_parent_regions_retained']
    views = data['proposal_views']
    trainviews = [info['frames'].index(f) for f in info['train_frames']]
    train = torch.tensor([int(v) in trainviews for v in views])
    weights = data['region_probability'].float().to(args.device)
    basis = payload['state_dict']['native_decoder.weight'].to(args.device)
    assert not payload['state_dict']['native_decoder.bias'].any()
    semantic_dim = basis.shape[1]
    descriptors = F.normalize(loaded['siglip']['descriptors'].float().to(args.device), dim=-1) @ basis
    old = payload['state_dict']['latent'][:, :semantic_dim].to(args.device)
    observed = payload['state_dict']['observed'].to(args.device)
    # Reproduce the actual parent first: region weights, not per-pixel weights.
    mass = weights[train].sum(0)
    reconstructed = weights[train].T @ descriptors[train] / mass[:, None].clamp_min(1e-8)
    torch.testing.assert_close(old[observed], reconstructed[observed], atol=2e-5, rtol=2e-4)
    results = {}
    for mode in (args.modes or ['queryfree', 'view_balanced']+(['view_origin_balanced'] if args.origin_balanced else [])):
        if mode == 'queryfree':
            selected = train & (torch.arange(len(views)) < original_count)
            w = weights[selected]
            mass = w.sum(0)
            estimate = w.T @ descriptors[selected] / mass[:, None].clamp_min(1e-8)
        elif mode=='view_balanced':
            estimate = torch.zeros_like(old)
            mass = torch.zeros(len(old), device=args.device)
            for vi in trainviews:
                selected = views == vi
                w = weights[selected]
                local = w.sum(0)
                estimate += w.T @ descriptors[selected] / local[:, None].clamp_min(1e-8)
                mass += (local > 1e-8)
            estimate /= mass[:, None].clamp_min(1e-8)
        else:
            from radio_gs.training.source_evidence_balancing import balance_source_views_and_origins
            origins=(torch.arange(len(views))>=original_count).long()
            group_views=views[train] if mode=='view_origin_balanced' else torch.zeros_like(views[train])
            estimate,mass=balance_source_views_and_origins(descriptors[train],weights[train],group_views.to(args.device),origins[train].to(args.device))
        valid = observed & (mass > 1e-8)
        state = {k: v.clone() for k, v in payload['state_dict'].items()}
        # Same support: retain original anchor where the control has no evidence.
        state['latent'][valid.cpu(), :semantic_dim] = estimate[valid].cpu()
        assert torch.equal(state['latent'][:, semantic_dim:], payload['state_dict']['latent'][:, semantic_dim:])
        name = 'source_fusion_' + mode + '_control'
        destination = root / name
        destination.mkdir(exist_ok=False)
        torch.save(dict(payload, state_dict=state), destination / 'field.pt')
        record = dict(report, schema='source_fusion_bias_control.v1',
                      field_sha256=sha256_file(destination / 'field.pt'),
                      parent_field_sha256=report['field_sha256'],
                      mode=mode, updated_elements=int(valid.sum()),
                      fallback_elements=int((observed & ~valid).sum()),
                      native_prefix_exactly_preserved=False, native_text_image_max_error=None,
                      structural_state_exactly_preserved=True, prompt_response_unchanged=True,
                      basis_unchanged=True, observed_unchanged=True,
                      inherited_calibration=True, original_region_count=original_count,
                      origin_balance_implementation_sha256=sha256_file('radio_gs/training/source_evidence_balancing.py') if mode in ('view_origin_balanced','origin_balanced') else None,
                      benchmark_data_used=False, implementation_sha256=sha256_file(__file__))
        (destination / 'report.json').write_text(json.dumps(record, indent=2) + '\n')
        results[name] = {k: record[k] for k in ['updated_elements', 'fallback_elements', 'field_sha256']}
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
