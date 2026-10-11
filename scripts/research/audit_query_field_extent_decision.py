"""Independent state/mask audit and fixed-scope objective attribution."""
from pathlib import Path
import time
import numpy as np
import torch
from numba import njit
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.extent_decision import ExtentDecision, render_row, CUTOFF
from scripts.research.run_query_field_extent_decision import ROOT, PARENT, FIELDS, ARMS, design
from scripts.research.run_query_field_relations import read, write, receipt, check
from scripts.research.summarize_query_field_relations import paired
from scripts.research.audit_query_field_extension_quality import compare_masks
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask


@njit
def render_all(pointer, index, alpha, selected):
    result = np.zeros(len(pointer)-1)
    for row in range(len(result)):
        result[row] = render_row(pointer, index, alpha, selected, row)
    return result


def objective(decoder, selected, evidence, qi, fixed):
    alpha = render_all(decoder.pointer, decoder.index, decoder.alpha, selected.numpy())
    alpha = alpha.clip(1e-6, 1-1e-6)
    probability = alpha*(1-CUTOFF)/(alpha*(1-CUTOFF)+CUTOFF*(1-alpha))
    support = .1 * np.dot(fixed['support_cost'].double().numpy(), selected.numpy())
    data = 0.
    for vi, confidence in zip(fixed['view_indices'], fixed['confidence']):
        a, b = decoder.slices[vi:vi+2]; p = probability[a:b]
        v = evidence['views'][vi]; y = v['target'][:, qi].double().numpy()
        pos = v['positive'][:, qi].numpy(); neg = v['negative'][:, qi].numpy(); known = pos | neg
        clipped = p.clip(1e-5, 1-1e-5)
        bce = -y*np.log(clipped)-(1-y)*np.log1p(-clipped)
        balanced = .5*((bce*pos).sum()/max(pos.sum(), 1)+(bce*neg).sum()/max(neg.sum(), 1))
        overlap = 1 - (2*(p*y*known).sum()+1)/((p*known).sum()+(y*known).sum()+1)
        data += confidence*(balanced+overlap)/len(fixed['view_indices'])
    return dict(total=float(data+support), data=float(data), support=float(support))


def main():
    d = design(); cfg = read(d['configuration']['path']); summary = read(ROOT/'summary.json')
    assert read(PARENT/'audit.json')['passed']
    old_receipts = {r['path']: r['sha256'] for r in read(PARENT/'design.json')['inputs']}
    for r in read(ROOT/'predictions_complete.json')['records']: check(r)
    scenes = {}
    for j in cfg['jobs']:
        out = ROOT/'predictions'/j['name']; parent = PARENT/'predictions'/j['name']
        record = read(out/'prediction.json'); check(dict(path=str(out/'predictions.pt'), sha256=record['predictions_sha256']))
        data = torch.load(out/'predictions.pt', weights_only=True); old = torch.load(parent/'predictions.pt', weights_only=True)
        field, manifest = load_field(FIELDS/j['scene']/'field')
        # Full cold reload and comparison of every pre-existing scene tensor.
        parent_state = torch.load(Path(j['field'])/'field.pt', weights_only=True)
        current = field.state_dict()
        for k, v in parent_state.items(): assert torch.equal(v, current[k]), k
        assert set(current)-set(parent_state) == {k for k in current if k.startswith('extent_factors.')}
        old_count = len(parent_state); del parent_state, current
        compilation = read(FIELDS/j['scene']/'compilation.json')
        assert compilation['field_sha256'] == manifest['files']['field.pt']['sha256']
        assert not compilation['queries_used'] and not compilation['scope_targets_in_export']
        for r in compilation['checks']:
            path = str(Path(j['observations'])/f"view_{r['frame']:05d}.pt")
            assert r['source_sha256'] == old_receipts[path]
            assert r['rays'] == 49152 and r['old_compiler_index_exact'] == 64
            assert all(c['max_alpha_error'] <= 1e-4 and c['selection_disagreements'] == 0 for c in r['cases'])
        for k, v in data['geometry'].items(): assert torch.equal(v, old['geometry'][k]) and torch.equal(v, getattr(field, k))
        assert torch.equal(data['observed'], old['observed'])
        for a, name in [('seed_addition', 'seed_addition'), ('dense_joint', 'admitted_joint'), ('source_initial', 'admitted_fusion')]:
            assert torch.equal(data['membership_by_control'][a], old['membership_by_control'][name])
        for a in ARMS:
            assert torch.equal(data['selected_by_control'][a], data['membership_by_control'][a] > .5)
            assert torch.equal(data['scores'][a], old['scores']['semantic_initial'])
            assert not data['selected_by_control'][a][~data['observed']].any()
        assert not record['target_labels_opened'] and not record['transient_query_fitting'] and not record['standard_deployment']
        evidence = torch.load(out/'privileged_scope.pt', weights_only=True)
        states = torch.load(parent/'admitted_optimization_state.pt', weights_only=True)
        decoder = ExtentDecision(field.extent_factors, len(field.xyz)); energies = []
        for qi, text in enumerate(data['texts']):
            if qi not in states:
                assert torch.equal(data['membership_by_control']['extent_discrete'][:, qi], old['membership_by_control']['semantic_initial'][:, qi])
                continue
            fixed = states[qi]
            assert torch.equal(fixed['start_membership'], data['membership_by_control']['source_initial'][:, qi])
            for vi, confidence, negative in zip(fixed['view_indices'], fixed['confidence'], fixed['negative']):
                v = evidence['views'][vi]
                assert float(v['confidence'][qi]) == confidence and torch.equal(v['negative'][:, qi], negative)
            value = {a: objective(decoder, data['selected_by_control'][a][:, qi], evidence, qi, fixed)
                     for a in ['source_initial', 'dense_joint', 'extent_discrete']}
            trace = record['diagnostics'][qi]['energy_trace']
            assert abs(trace[0]-value['source_initial']['total']) < 1e-7
            assert abs(trace[-1]-value['extent_discrete']['total']) < 1e-7
            assert np.all(np.diff(trace) <= 1e-8)
            energies.append(dict(query=text, values=value))
        del field, decoder, old
        evaluation = read(out/'evaluation/evaluation.json'); metrics = {}; masks = {}; objects = {}
        for a in ARMS:
            rows = evaluation['metrics'][a]['selected']['objects']; objects[a] = {(r['frame'], r['query']): r for r in rows}
            assert len(rows) == len(objects[a]); masks[a] = {}; ious = []; precisions = []; recalls = []
            for key, r in objects[a].items():
                assert not r['missing']
                gt = load_binary_mask(Path(r['gt_path']), threshold=10)
                pred = load_binary_mask(Path(r['pred_path']), threshold=10, grayscale=True)
                tp = int((gt & pred).sum()); iou = tp/max(int((gt | pred).sum()), 1)
                assert abs(iou-r['iou']) < 1e-12
                ious.append(iou); precisions.append(tp/max(int(pred.sum()), 1)); recalls.append(tp/max(int(gt.sum()), 1))
                masks[a][key] = pred
            assert abs(float(np.mean(ious))-summary['per_scene'][a][j['scene']]) < 1e-12
            metrics[a] = dict(iou=float(np.mean(ious)), precision=float(np.mean(precisions)), recall=float(np.mean(recalls)))
        changes = {}
        for base in ['seed_addition', 'dense_joint', 'source_initial']:
            assert objects[base].keys() == objects['extent_discrete'].keys(); rows = []
            for key, r in objects[base].items():
                assert r['gt_path'] == objects['extent_discrete'][key]['gt_path']
                gt = load_binary_mask(Path(r['gt_path']), threshold=10)
                rows.append(dict(frame=key[0], query=key[1], **compare_masks(masks[base][key], masks['extent_discrete'][key], gt)))
            changes[base] = dict(**{k: sum(r[k] for r in rows) for k in ['added_tp','added_fp','removed_tp','removed_fp']},
                beneficial_cases=sum(r['beneficial'] for r in rows), harmful_cases=sum(r['harmful'] for r in rows), rows=rows)
        scenes[j['scene']] = dict(metrics=metrics, changes=changes, objective_comparisons=energies,
            parent_tensors_exact=old_count, compilation=compilation, active_queries=len(states),
            decisions=record['diagnostics'])
        print('audited', j['scene'], flush=True)
    macro = {a: {k: sum(r['metrics'][a][k] for r in scenes.values())/4 for k in ['iou','precision','recall']} for a in ARMS}
    comparisons = {a: paired({s: r['metrics'][a]['iou'] for s,r in scenes.items()},
        {s: r['metrics']['extent_discrete']['iou'] for s,r in scenes.items()}, 'actual selected-set IoU') for a in ['seed_addition','dense_joint','source_initial']}
    result = dict(passed=True, macro=macro, comparisons=comparisons, scenes=scenes,
        design=receipt(ROOT/'design.json'), code=receipt(__file__), all_ious_replayed=True,
        whole_extent_objectives_replayed=True, privileged_scopes=True, standard_deployment=False,
        default_promoted=False, whole_five_task=False)
    write(ROOT/'audit.json', result); write('docs/experiments/2026-10-11-extent-decision-audit.json', result)
    print(macro, flush=True); print(comparisons, flush=True)


if __name__ == '__main__': torch.set_num_threads(4); main()
