"""Read-only source-mask audit of direct gradients in the retained scope head."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
from radio_gs.utils.immutable_artifacts import sha256_file


def counterexample():
    head = GroundedSemanticTransfer(2, 1, 2, 2)
    with torch.no_grad():
        head.prototypes.copy_(torch.eye(2)); head.acceptance.zero_()
        head.decoder.weight.zero_(); head.decoder.bias.copy_(torch.tensor([-.2, 1.2]))
    prediction = head(torch.eye(2), torch.tensor([[1., 1.]]))[0].sigmoid()
    target = torch.tensor([[1., 0.]])
    F.binary_cross_entropy(prediction, target).backward()
    return dict(raw=[-.2, 1.2], target=target.tolist(), prediction=prediction.detach().tolist(),
                decoder_bias_gradient=head.decoder.bias.grad.tolist(),
                shared_parameter_updates_can_still_move_saturated_elements=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    field_path = args.root/'bounded_fallback_admission/field.pt'
    payload = torch.load(field_path, map_location='cpu', weights_only=False)
    assert sha256_file(field_path) == json.loads(field_path.with_name('report.json').read_text())['field_sha256']
    model = ElementRetrievalField(**payload['architecture']).to(args.device)
    model.load_state_dict(payload['state_dict']); model.requires_grad_(False)
    head = model.grounded_transfer
    assert not head.hidden_dim and head.identity_selection['completion_mode'] == 'native_bounded_extent'
    assert not head.identity_selection.get('calibrate_accepted_concepts', False)
    evidence_path = args.root/'caption_region_evidence.pt'
    evidence = torch.load(evidence_path, map_location='cpu', weights_only=False)
    assert evidence['source_only'] and not evidence['benchmark_data_used']
    assert evidence['geometry_binding'] == payload['geometry_binding']
    info = json.loads((args.root/'teachers.json').read_text())
    assert evidence['teacher_manifest_sha256'] == sha256_file(args.root/'teachers.json')
    authority = json.loads((args.root/'responsibility.json').read_text())
    assert info['authority_sha256'] == sha256_file(args.root/'responsibility.json')
    selected_frames = info['train_frames'][:2] + info['train_frames'][-2:]
    examples = []
    # Source order, confidence/area policy and quota are fixed before model reads.
    for frame in selected_frames:
        targets = {}
        for i, record in enumerate(evidence['records']):
            if record['frame'] != frame or record['confidence'] < .7 or int(evidence['masks'][i].sum()) < 20:
                continue
            for qi in record['phrase_indices']:
                targets[qi] = targets.get(qi, torch.zeros_like(evidence['masks'][i], dtype=torch.bool)) | evidence['masks'][i].bool()
        seen = set()
        for qi, target in targets.items():
            key = target.numpy().tobytes()
            if key in seen: continue
            seen.add(key)
            examples.append(dict(frame=frame, qi=qi, target=target.flatten()))
            if len(seen) == 8: break
    query_ids = list(dict.fromkeys(row['qi'] for row in examples))
    queries = evidence['text_embeddings'][query_ids].float().to(args.device)
    captured = {}
    def capture_head(module, inputs, outputs):
        captured['proposal_logits'] = outputs[0].detach().clone()
        captured['proposal_accepted'] = outputs[1].detach().clone()
    def capture_admission(module, inputs, outputs):
        captured['benefit_accepted'] = outputs.detach().clone()
    handles = [head.register_forward_hook(capture_head), model.extent_benefit_admission.register_forward_hook(capture_admission)]
    with torch.no_grad():
        all_index, all_concept, _ = head.route(evidence['text_embeddings'].float().to(args.device))
        concept_index, concept_accepted, _ = head.route(queries)
        probability = model.logits(queries, 'text').sigmoid() * model.observed[:, None]
        active = concept_accepted & captured['proposal_accepted'] & captured['benefit_accepted']
        features = head.features(model.latent)
        columns = concept_index + head.identity_selection['trusted_concept_start']
        raw = F.linear(features, head.decoder.weight[columns], head.decoder.bias[columns])
        if head.residual is not None:
            hidden = head.residual[1](head.residual[0](features))
            raw += F.linear(hidden, head.residual[2].weight[columns], head.residual[2].bias[columns])
        replay = float((probability[:, active] - raw[:, active].clamp(1e-6, 1-1e-6)*model.observed[:, None]).abs().max()) if active.any() else None
        assert replay is None or replay < 2e-6
    for handle in handles: handle.remove()
    results = []; hits = {}; h, w = evidence['masks'].shape[-2:]
    with torch.no_grad():
        for example in examples:
            frame = example['frame']; col = query_ids.index(example['qi'])
            if frame not in hits:
                record = authority['views'][info['frames'].index(frame)]
                path = args.root/record['relative_path']; assert sha256_file(path) == record['sha256']
                hit = torch.load(path, map_location=args.device, weights_only=False)
                pid, gid, weight = hit['pixel_ids'].long(), hit['gaussian_ids'].long(), hit['base_weights'].float()
                full = torch.zeros(h*w, device=args.device).scatter_add_(0, pid, weight)
                keep = model.observed[gid]; pid, gid, weight = pid[keep], gid[keep], weight[keep]
                mass = torch.zeros_like(full).scatter_add_(0, pid, weight)
                valid = (mass/full.clamp_min(1e-8) > .5) & (full > 1e-8)
                hits[frame] = pid, gid, weight/mass[pid].clamp_min(1e-8), valid
            pid, gid, normalized, valid = hits[frame]
            def render(values): return torch.zeros(h*w, device=args.device).scatter_add_(0, pid, normalized*values[gid])
            pred = render(probability[:, col]); target = example['target'].to(args.device)
            positive = target & valid; negative = ~target & valid
            fn = positive & (pred <= .5); fp = negative & (pred > .5)
            row = dict(frame=frame, query_index=example['qi'], phrase=evidence['phrases'][example['qi']],
                       source_positive_pixels=int(target.sum()), supported_positive_pixels=int(positive.sum()),
                       supported_negative_pixels=int(negative.sum()), false_negative_pixels=int(fn.sum()),
                       false_positive_pixels=int(fp.sum()), concept_accepted=bool(concept_accepted[col]),
                       proposal_accepted=bool(captured['proposal_accepted'][col]),
                       benefit_accepted=bool(captured['benefit_accepted'][col]),
                       active_differentiable_concept=bool(active[col]))
            if active[col]:
                low = render((raw[:, col] < 1e-6).float()); high = render((raw[:, col] > 1-1e-6).float())
                gate_low = render((raw[:, col] <= .25).float()); gate_high = render((raw[:, col] >= .75).float())
                for name, mask, values in [('positive_low_clamped_mass', positive, low), ('false_negative_low_clamped_mass', fn, low),
                                           ('negative_high_clamped_mass', negative, high), ('false_positive_high_clamped_mass', fp, high),
                                           ('false_negative_low_selected_gate_mass', fn, gate_low), ('false_positive_high_selected_gate_mass', fp, gate_high)]:
                    row[name] = dict(sum=float(values[mask].sum()), pixels=int(mask.sum()), mean=float(values[mask].mean()) if mask.any() else None)
                row['positive_error_mass'] = float((1-pred[positive]).sum())
                row['negative_error_mass'] = float(pred[negative].sum())
                row['false_negative_low_clamped_majority_pixels'] = int((fn & (low > .5)).sum())
                row['false_positive_high_clamped_majority_pixels'] = int((fp & (high > .5)).sum())
                row['all_observed_low_clamped_fraction'] = float((raw[model.observed, col] < 1e-6).float().mean())
                row['all_observed_high_clamped_fraction'] = float((raw[model.observed, col] > 1-1e-6).float().mean())
            results.append(row)
    active_rows = [r for r in results if r['active_differentiable_concept']]
    summary = dict(examples=len(results), unique_queries=len(query_ids), active_concept_examples=len(active_rows),
                   concept_accepted_unique_queries=int(concept_accepted.sum()), active_unique_queries=int(active.sum()),
                   all_evidence_phrases=len(all_concept), all_evidence_concept_accepted=int(all_concept.sum()),
                   direct_concept_probability_replay_max_error=replay)
    for key in ('supported_positive_pixels', 'supported_negative_pixels', 'false_negative_pixels', 'false_positive_pixels',
                'false_negative_low_clamped_majority_pixels', 'false_positive_high_clamped_majority_pixels',
                'positive_error_mass', 'negative_error_mass'):
        summary['active_'+key] = sum(r[key] for r in active_rows)
    for key in ('positive_low_clamped_mass', 'false_negative_low_clamped_mass', 'negative_high_clamped_mass', 'false_positive_high_clamped_mass',
                'false_negative_low_selected_gate_mass', 'false_positive_high_selected_gate_mass'):
        total = sum(r[key]['sum'] for r in active_rows); count = sum(r[key]['pixels'] for r in active_rows)
        summary[key] = dict(sum=total, pixels=count, mean=total/count if count else None)
    report = dict(schema='source_scope_gradient_audit.next32.v1', source_only=True, benchmark_data_read=False,
                  cohort='First2 and last2 registered training views; first8 distinct confidence>=.7 area>=20 phrase-union scopes per view, source insertion order; fixed before outcomes.',
                  interpretation='These are source pseudo-label conflicts, not verified benchmark errors. Per-element direct gradient zero does not prevent shared parameter updates moving the element. Hard native fallback is an intended semantic-protection inference path, not an end-to-end learned selector.',
                  root=str(args.root), implementation_sha256=sha256_file(__file__),
                  bindings={str(p):sha256_file(p) for p in [field_path, evidence_path,args.root/'teachers.json',args.root/'responsibility.json',
                            Path('radio_gs/models/grounded_semantic_transfer.py'),Path('radio_gs/models/element_retrieval_field.py')]},
                  architecture=payload['architecture'], counterexample=counterexample(), summary=summary, examples=results)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
