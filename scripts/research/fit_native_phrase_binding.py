"""Matched native-phrase binding and query-independent witness control."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.querying.native_phrase_binding import NativePhraseBinding, spatial_witnesses
from radio_gs.querying.referent_identity import native_logit_scale
from scripts.research.fit_referent_grounder import read_scene
from scripts.research.prepare_native_phrase_binding import ROOT, write
from scripts.research.prepare_referent_grounding import ENCODER


def load_scene(scene):
    data = read_scene(scene, 'cuda', ROOT)
    phrases = np.load(ROOT / 'phrases' / (scene + '.npz'))
    rows = json.loads((ROOT / 'phrases' / (scene + '.json')).read_text())
    pooled = torch.from_numpy(phrases['pooled'].astype(np.float32)).cuda()
    pooled = F.normalize(pooled, dim=-1)
    scores = pooled @ data['visual'].T
    spatial = spatial_witnesses(data['geometry'], scores, native_logit_scale(ENCODER))
    data['native'] = data['pooled'] @ data['visual'].T
    data['phrase_rows'] = rows;data['queries'] = []
    for i, row in enumerate(rows):
        context = torch.stack([data['tokens'][i, m['tokens']].mean(0) for m in row]) if row else data['tokens'].new_empty((0, 1152))
        sl = slice(phrases['offsets'][i], phrases['offsets'][i + 1])
        data['queries'].append(dict(geometry=data['geometry'], native=data['native'][i],
            phrase_scores=scores[sl], spatial={key: value[sl] for key, value in spatial.items()},
            phrase_tokens=context, roles=torch.tensor([m['role'] for m in row], device='cuda'),
            query_tokens=data['tokens'][i, data['valid'][i]].mean(0)))
    return data


def instance_competitors(data):
    """Training-only distributions over regional observations of each instance.

Normalize each instance separately: many redundant views do not create more
negative labels. Official semantic classes define homonymous competitors.
"""
    scene = data['scene']
    targets = np.load(ROOT / 'entities' / (scene + '.npz'))
    annotations = json.loads((Path('output/referent_grounding_20261005/raw') / scene /
        (scene + '.aggregation.json')).read_text())['segGroups']
    categories = {int(g['objectId']) + 1: g['label'] for g in annotations}
    instances = targets['instances']
    purity = targets['purity'][data['ids']].T ** 4
    supported = purity.sum(1) > 0
    purity /= np.maximum(purity.sum(1, keepdims=True), 1e-30)
    distributions = torch.from_numpy(purity).cuda()
    result = []
    for query in data['rows']:
        target = query['raw_object_id'] + 1
        competitors = [j for j, instance in enumerate(instances)
            if supported[j] and categories.get(int(instance)) == categories[target]]
        desired = [j for j, column in enumerate(competitors) if instances[column] == target]
        result.append((distributions[competitors], desired[0]) if len(competitors) > 1 and desired else None)
    return result


@torch.inference_mode()
def evaluate(model, scenes, witness, interventions=False):
    model.eval();records = []
    for data in scenes:
        native = data['native'] * model.native_scale
        scores = [];uniform = [];reversed_scores = [];target_only = []
        for query in data['queries']:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                scores.append(model(**query, witness=witness))
                if interventions:
                    uniform.append(model(**query, witness='uniform'))
                    reversed_scores.append(model(**query, witness='reversed'))
            target = query['roles'] == 1
            target_only.append(query['phrase_scores'][target].amax(0) if target.any() else query['native'])
        learned = torch.stack(scores);delta = learned - native
        common = (delta - delta.mean(1, keepdim=True)).mean(0, keepdim=True)
        conditions = dict(original=learned, native=native, predicted_target=torch.stack(target_only))
        if interventions:
            conditions.update(query_mean=native + common, uniform=torch.stack(uniform), reversed=torch.stack(reversed_scores))
        metrics = {};row = torch.arange(len(scores), device='cuda')
        for key, logits in conditions.items():
            seed = logits.argmax(1)
            metrics[key] = dict(iou=float(data['reward'][row, seed].mean()),
                majority=float((data['purity'][row, seed] > .5).float().mean()),
                per_query_iou=data['reward'][row, seed].cpu().tolist(),
                seeds=data['ids'][seed.cpu().numpy()].tolist())
        records.append(dict(scene=data['scene'], queries=len(scores), conditions=metrics))
    return dict(scenes=records, conditions={key: {metric: float(np.mean([r['conditions'][key][metric] for r in records]))
        for metric in ['iou', 'majority']} for key in records[0]['conditions']})


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--condition', choices=['native', 'uniform'], required=True)
    p.add_argument('--instance-objective', action='store_true')
    args = p.parse_args();torch.set_num_threads(3);torch.manual_seed(20261006);np.random.seed(20261006)
    design = json.loads((ROOT / 'design.json').read_text())
    groups = {key: [load_scene(s) for s in design[key]] for key in ['fit', 'calibration', 'validation']}
    if args.instance_objective:
        for data in groups['fit']:
            data['competitors'] = instance_competitors(data)
    architecture = dict(width=64, text_width=1152, native_scale=native_logit_scale(ENCODER))
    model = NativePhraseBinding(**architecture).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    name = args.condition + ('_instance' if args.instance_objective else '')
    rng = np.random.default_rng(20261006);out = ROOT / name;out.mkdir(exist_ok=True)
    eligible = [];counts = []
    for data in groups['fit']:
        listener = [j for j, row in enumerate(data['rows']) if row['listener_correct'] and float(data['reward'][j].max()) > 0]
        eligible.append(np.array([j for j in listener if data['rows'][j]['mentions_target_class']]))
        counts.append(len(listener))
    def save(epoch):
        torch.save(dict(model_type='native_phrase_binding', architecture=architecture,
            state_dict={k: v.cpu() for k, v in model.state_dict().items()}, epoch=epoch, witness=args.condition,
            role_checkpoint=str((ROOT / 'roles/model.pt').resolve()),
            training_condition='disjoint real RGB-D rooms; NR3D, official3D instances and ScanEnts3D roles',
            instance_objective=args.instance_objective,
            fit=design['fit'], calibration=design['calibration']), out / 'model.pt')
    start = time.monotonic();initial = evaluate(model, groups['calibration'], args.condition)
    best = initial['conditions']['original']['iou'];save(0);history = [dict(epoch=0, calibration=initial)]
    for epoch in range(1, 13):
        tasks = []
        for i, ids in enumerate(eligible):
            shuffled = ids.copy();rng.shuffle(shuffled);shuffled = np.resize(shuffled, counts[i])
            tasks.extend((i, shuffled[j:j + 8]) for j in range(0, len(shuffled), 8))
        rng.shuffle(tasks);model.train();losses = [];relation_losses = []
        for scene_index, ids in tasks:
            data = groups['fit'][scene_index];optimizer.zero_grad(set_to_none=True);accumulated = 0.
            for index in ids:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    scores, relation = model(**data['queries'][index], witness=args.condition, return_components=True)
                target = data['reward'][index].pow(4);target = target / target.sum()
                main_loss = -(target * F.log_softmax(scores.float(), dim=-1)).sum()
                relation_loss = scores.sum() * 0
                if args.instance_objective and (data['queries'][index]['roles'] == 2).any() and data['competitors'][index] is not None:
                    distribution, desired = data['competitors'][index]
                    # Supervise the relation output. Its language layer is
                    # shared with local identity, so this is not gradient
                    # isolation of the semantic branch (see typed binding).
                    instance_scores = distribution @ relation.float()
                    relation_loss = -F.log_softmax(instance_scores, dim=0)[desired]
                    relation_losses.append(float(relation_loss))
                loss = (main_loss + relation_loss) / len(ids)
                loss.backward();accumulated += float(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.);optimizer.step();losses.append(accumulated)
        calibration = evaluate(model, groups['calibration'], args.condition);value = calibration['conditions']['original']['iou']
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)),
            relation_loss=float(np.mean(relation_losses)) if relation_losses else None, calibration=calibration))
        if value > best:
            best = value;save(epoch)
        print(name, epoch, 'loss', round(np.mean(losses), 4), 'calibration', round(value, 5), 'best', round(best, 5), flush=True)
    saved = torch.load(out / 'model.pt', map_location='cuda', weights_only=False);model.load_state_dict(saved['state_dict'])
    report = dict(condition=name, architecture=architecture, selected_epoch=saved['epoch'], epochs=12,
        instance_objective=args.instance_objective,
        instance_supervised_training_queries=sum(sum(data['competitors'][i] is not None and
            bool((data['queries'][i]['roles'] == 2).any()) for i in ids)
            for data, ids in zip(groups['fit'], eligible)) if args.instance_objective else 0,
        eligible_records=sum(map(len, eligible)), presentations_per_epoch=sum(counts),
        parameter_count=sum(p.numel() for p in model.parameters()), history=history,
        calibration=evaluate(model, groups['calibration'], args.condition, True),
        validation=evaluate(model, groups['validation'], args.condition, True),
        seconds=time.monotonic() - start, native_coordinates_changed=False, scope_changed=False, benchmark_data_used=False,
        deployed_phrases='predicted by query-only role extractor; no ground-truth phrase spans')
    write(out / 'fit.json', report)
    print('complete', name, report['validation']['conditions'], flush=True)


if __name__ == '__main__':
    main()
