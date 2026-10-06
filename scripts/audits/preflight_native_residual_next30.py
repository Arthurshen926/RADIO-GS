"""Source-view-only compression admission across the four requested protocols."""
import argparse
import json
from pathlib import Path
import sys

import torch

from radio_gs.training.query_preserving_basis import native_response_diagnostic
from radio_gs.training.native_semantic_residual import fit_native_residual
from radio_gs.utils.immutable_artifacts import sha256_file


def cohort():
    rows = []
    for scene in ['figurines', 'ramen', 'teatime', 'waldo_kitchen']:
        rows.append(dict(key='lerf_'+scene, benchmark='lerf', scene=scene,
            root=str(Path('output/full_field_rebuild_next19')/scene/'reconstruction/field'), run='bounded_fallback_admission'))
    adaptive = json.loads(Path('output/scannet_expanded_budget_next29/adaptive_plan.json').read_text())
    for scene in adaptive['scenes']:
        if scene in adaptive['rebuild_scenes']:
            root = Path('output/scannet_expanded_budget_next29')/scene/'reconstruction/field'
        else:
            root = (Path('output/scannet_coverage_native_probe_20260919') if scene == 'scene0062_00'
                    else Path('output/scannet_coverage_native_next28'))/scene/'reconstruction_complete_next28/field'
        rows.append(dict(key='scannet_'+scene, benchmark='scannet', scene=scene, root=str(root), run='bounded_fallback_admission'))
    for scene in ['flower', 'fortress', 'fern', 'horns_center', 'leaves', 'orchids', 'trex']:
        rows.append(dict(key='nvos_'+scene, benchmark='nvos', scene=scene,
            root=str(Path('output/nvos_ropefix_pipeline_20260917')/scene/'field'),
            run='image_supervised_native_residual_measured_support512'))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--keys', nargs='+')
    p.add_argument('--register', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    out = Path('output/native_residual_basis_next30')
    if a.register:
        out.mkdir(exist_ok=False)
        record = dict(cohort=cohort(), dimensions=[32, 64, 128], weights=[0., 1.],
            selection='Smallest residual capacity reducing both text and image heldout cosine MSE by at least 20%; centered MSE must not rise and teacher top1 agreement may drop at most 1pp; minimize worst MSE ratio within admitted capacity.',
            image_query_sampling='512 evenly spaced training regions; never validation regions',
            benchmark_queries_used=False, target_gt_used=False, independent_test=False,
            nvos_horns_left_shares_map='horns_center',
            implementation_sha256=sha256_file(__file__),
            basis_implementation_sha256=sha256_file('radio_gs/training/native_semantic_residual.py'), diagnostic_implementation_sha256=sha256_file('radio_gs/training/query_preserving_basis.py'))
        (out/'registration.json').write_text(json.dumps(record, indent=2)+'\n')
        print([r['key'] for r in record['cohort']])
        return
    record = json.loads((out/'registration.json').read_text())
    assert record['implementation_sha256'] == sha256_file(__file__)
    assert record['basis_implementation_sha256'] == sha256_file('radio_gs/training/native_semantic_residual.py')
    assert record['diagnostic_implementation_sha256'] == sha256_file('radio_gs/training/query_preserving_basis.py')

    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if '.labels.ply' in path or '/lerf_ovs/label/' in path or '/evaluation/report.json' in path:
                raise RuntimeError('Target evidence forbidden during basis selection')
    sys.addaudithook(guard)

    for row in record['cohort']:
        if a.keys and row['key'] not in a.keys:
            continue
        root, dest = Path(row['root']), out/row['key']
        dest.mkdir(exist_ok=False)
        info = json.loads((root/'teachers.json').read_text())

        def load(path, digest):
            assert sha256_file(path) == digest
            return torch.load(path, map_location='cpu', weights_only=False)

        tr, sr = info['teacher_files']['siglip'], info['teacher_files']['sam']
        teacher = load(root/tr['path'], tr['sha256'])
        sam = load(root/sr['path'], sr['sha256'])
        views = sam['view_indices']
        train = torch.tensor([info['frames'][int(v)] in info['train_frames'] for v in views])
        validation = torch.tensor([info['frames'][int(v)] in info['validation_frames'] for v in views])
        assert not (train & validation).any() and (train | validation).all()
        x = teacher['descriptors'][train].float().cuda()
        held = teacher['descriptors'][validation].float().cuda()
        text = teacher['text_embeddings'].float().cuda()
        image = x[torch.linspace(0, len(x)-1, min(512, len(x))).long()]
        fr = json.loads((root/row['run']/'report.json').read_text())
        current = load(root/row['run']/'field.pt', fr['field_sha256'])['state_dict']['native_decoder.weight'].T.cuda()

        def assess(basis):
            return {name: native_response_diagnostic(held, q, basis) for name, q in [('text', text), ('image', image)]}

        baseline = assess(current)
        candidates = []
        best, best_ratio, best_weight, best_dim = current, 1., None, 0
        for dimensions in record['dimensions']:
            for weight in record['weights']:
                residual = fit_native_residual(x, text, current, dimensions, weight)
                basis = torch.cat([current, residual])
                metrics = assess(basis)
                ratios = {m: metrics[m]['mse']/max(baseline[m]['mse'], 1e-12) for m in ['text', 'image']}
                ratio = max(ratios.values())
                admissible = ratio <= .8 and all(metrics[m]['centered_mse'] <= baseline[m]['centered_mse'] and metrics[m]['top1_agreement'] >= baseline[m]['top1_agreement'] - .01 for m in ['text', 'image'])
                candidates.append(dict(dimensions=dimensions, weight=weight, metrics=metrics, mse_ratios=ratios, admissible=admissible))
                if admissible and ratio < best_ratio:
                    best, best_ratio, best_weight, best_dim = basis, ratio, weight, dimensions
            if best_dim:
                break
        torch.save(dict(basis=best.cpu(), dimensions=len(best), residual_dimensions=best_dim), dest/'basis.pt')
        result = dict(**row, source_only=True, target_gt_used=False, benchmark_queries_used=False,
            teacher_manifest_sha256=sha256_file(root/'teachers.json'), parent_field_sha256=fr['field_sha256'],
            training_regions=len(x), validation_regions=len(held), train_frames=info['train_frames'],
            validation_frames=info['validation_frames'], baseline=baseline, candidates=candidates,
            residual_dimensions=best_dim, selected_weight=best_weight, source_admitted=best_weight is not None,
            worst_mse_ratio=best_ratio, selected_basis_sha256=sha256_file(dest/'basis.pt'),
            registration_sha256=sha256_file(out/'registration.json'))
        (dest/'report.json').write_text(json.dumps(result, indent=2)+'\n')
        print(row['key'], 'selected', best_weight, 'worst ratio', best_ratio, flush=True)


if __name__ == '__main__':
    main()
