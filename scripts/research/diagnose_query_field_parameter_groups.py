"""Post-score factorial parameter intervention on all figurines queries.

Local failure attribution only: hybrid states are not deployed or promoted.
The group codes and the shared scale/affinity parameters are swapped as blocks.
"""
import gc
from pathlib import Path
from types import SimpleNamespace

import torch

from radio_gs.query_field.contracts import QueryPacket
from radio_gs.query_field.storage import load_field
from scripts.research.run_query_field_relations import ROOT, read, write, receipt, check


@torch.no_grad()
def main():
    out = ROOT / 'diagnostics/figurines_parameter_groups'
    out.mkdir(parents=True, exist_ok=False)
    base = next(j for j in read('configs/query_field/five_benchmark_v1/cohort.json')['lerf'] if j['scene'] == 'figurines')
    current = next(j for j in read(ROOT / 'cross_view_anchor/cohort.json')['lerf'] if j['scene'] == 'figurines')
    names = ['group', 'scale_cutoffs', 'log_temperature', 'affinity_bias']
    jobs = {'new_group_only': {'group'}, 'new_shared_parameters_only': set(names) - {'group'}}
    design = dict(code=receipt(Path(__file__)), parent=receipt(Path(base['field']) / 'field.pt'),
        current=receipt(Path(current['field']) / 'field.pt'), interventions={k: sorted(v) for k, v in jobs.items()},
        selected_after_full_scores=True, scene_selection='largest actual-set regression',
        diagnostic_only=True, model_promotion=False)
    write(out / 'design.json', design)
    parent = torch.load(design['parent']['path'], map_location='cpu', weights_only=True)
    parent = {k: parent[k] for k in names}; gc.collect()
    field, _ = load_field(current['field'], 'cuda')
    updated = {k: p.detach().cpu().clone() for k, p in field.named_parameters()}
    if set(updated) != set(names): raise ValueError('trainable parameter blocks changed')
    source_path = Path(base['predictions']) / 'predictions.pt'
    check(dict(path=str(source_path), sha256=read(Path(base['predictions']) / 'prediction.json')['predictions_sha256']))
    reference_path = Path(current['predictions']) / 'predictions.pt'
    check(dict(path=str(reference_path), sha256=read(Path(current['predictions']) / 'prediction.json')['predictions_sha256']))
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    # The unmodified current field must replay before any hybrid is evaluated.
    for i, vector in enumerate(source['native_query_vectors']):
        result = field.query(QueryPacket('text', task='union', semantic=vector))
        if not torch.equal(result.membership_scores.cpu(), reference['scores']['shared_membership'][:, i]):
            raise ValueError('current full parameter state does not replay exactly')
    predictions = []
    for name, changed in jobs.items():
        for key, parameter in field.named_parameters():
            parameter.copy_((updated if key in changed else parent)[key])
        values, selected = [], []
        for vector in source['native_query_vectors']:
            result = field.query(QueryPacket('text', task='union', semantic=vector))
            values.append(result.membership_scores.cpu()); selected.append(result.selected.cpu())
        path = out / (name + '.pt')
        torch.save(dict(scores=torch.stack(values, 1), selected=torch.stack(selected, 1)), path)
        predictions.append(receipt(path))
    write(out / 'predictions_complete.json', dict(records=predictions, design=receipt(out / 'design.json')))
    # Both interventions are complete before their scores are opened.
    from radio_gs.evaluation.lerf_readouts import evaluate_lerf_readouts
    scene_path = read(Path(base['predictions']) / 'prediction.json')['job']['evaluation_scene']
    scene = read(scene_path)
    geometry = SimpleNamespace(**{k: v.cuda() for k, v in source['geometry'].items()})
    observed = source['observed'].cuda()
    results = {}
    for name in jobs:
        p = torch.load(out / (name + '.pt'), weights_only=True)
        directory = out / name; directory.mkdir()
        results[name] = evaluate_lerf_readouts(scene='figurines', out=directory, texts=source['texts'],
            scores={'shared_membership': p['scores'].cuda()}, selection_scores={'shared_membership': p['selected'].float().cuda()},
            geo=geometry, default_observed=observed, semantic_geo=geometry, semantic_observed=observed,
            selection_geo=geometry, selection_observed=observed, camera_report=scene['camera_report'],
            camera_report_sha256=scene['camera_report_sha256'])
        print(name, results[name]['shared_membership']['iou'], results[name]['shared_membership']['selected']['miou'], flush=True)
    write(out / 'evaluation.json', dict(results=results, parent_parameters={k: v.tolist() for k, v in parent.items() if k != 'group'},
        new_parameters={k: v.tolist() for k, v in updated.items() if k != 'group'}, diagnostic_only=True,
        all_current_queries_replayed_exactly=True, inputs=[receipt(source_path), receipt(reference_path)],
        predictions=receipt(out / 'predictions_complete.json'), design=receipt(out / 'design.json')))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
