"""Typed semantic supervision followed by isolated instance relation learning."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.querying.typed_referent_binding import TypedReferentBinding
from radio_gs.querying.referent_identity import native_logit_scale
from scripts.research.fit_native_phrase_binding import load_scene
from scripts.research.prepare_native_phrase_binding import ROOT as SOURCE, write
from scripts.research.prepare_referent_grounding import ENCODER

ROOT = Path('output/typed_referent_binding_20261006')


def semantic_targets(data):
    scene = data['scene']
    entities = np.load(SOURCE / 'entities' / (scene + '.npz'))
    groups = json.loads((Path('output/referent_grounding_20261005/raw') / scene /
        (scene + '.aggregation.json')).read_text())['segGroups']
    categories = {int(g['objectId']) + 1: g['label'] for g in groups}
    instances = entities['instances'];purity = entities['purity'][data['ids']].T
    known = purity.sum(0)
    # Each same-class instance contributes equal positive mass regardless of
    # its number of repeated regional observations. Unlabelled mass is unknown.
    weight = purity ** 4
    weight /= np.maximum(weight.sum(1, keepdims=True), 1e-30)
    desired = [];classes = [];dominant = purity.argmax(0)
    types = [];dominant_mass = purity.max(0)
    for query in data['rows']:
        target = query['raw_object_id'] + 1
        columns = [i for i, instance in enumerate(instances) if categories.get(int(instance)) == categories[target]]
        positive = weight[columns].sum(0);positive /= np.maximum(positive.sum(), 1e-30)
        desired.append(positive);classes.append(purity[columns].sum(0))
        kind = np.full(len(data['ids']), 2, np.int64)
        kind[dominant_mass <= .5] = 3
        for column in columns:
            kind[(dominant == column) & (dominant_mass > .5)] = 0 if instances[column] == target else 1
        types.append(kind)
    data['class_target'] = torch.from_numpy(np.stack(desired)).cuda()
    data['class_fraction'] = torch.from_numpy(np.stack(classes)).cuda()
    data['known_fraction'] = torch.from_numpy(known).cuda()
    data['identity_types'] = np.stack(types)


@torch.inference_mode()
def evaluate(model, scenes, interventions=False):
    model.eval();records = []
    for data in scenes:
        original = [];identity = [];deltas = [];reversed_scores = [];full_weights = []
        for query in data['queries']:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                scores, prior, delta = model(query, return_components=True)
                _, weights = model.identity(query, return_weights=True)
                if interventions:
                    reversed_scores.append(model(query, witness='reversed'))
            original.append(scores);identity.append(prior);deltas.append(delta);full_weights.append(float(weights[0]))
        prior = torch.stack(identity);change = torch.stack(deltas)
        conditions = dict(original=torch.stack(original), identity=prior,
            native=data['native'] * model.identity.native_scale)
        if interventions:
            conditions.update(query_mean=prior + (change - change.mean(1, keepdim=True)).mean(0, keepdim=True),
                reversed=torch.stack(reversed_scores))
        metrics = {};row = torch.arange(len(data['rows']), device='cuda')
        for key, score in conditions.items():
            seed = score.argmax(1);seed_cpu = seed.cpu().numpy()
            counts = np.bincount(data['identity_types'][np.arange(len(seed_cpu)), seed_cpu], minlength=4)
            metrics[key] = dict(iou=float(data['reward'][row, seed].mean()),
                class_accuracy=float((data['class_fraction'][row, seed] > .5).float().mean()),
                instance_accuracy=float((data['purity'][row, seed] > .5).float().mean()),
                errors=dict(zip(['correct_instance', 'same_class_wrong_instance', 'wrong_class', 'mixed_region'], counts.tolist())),
                per_query_iou=data['reward'][row, seed].cpu().tolist(),
                seeds=data['ids'][seed_cpu].tolist())
        records.append(dict(scene=data['scene'], queries=len(data['rows']), conditions=metrics,
            mean_full_query_weight=float(np.mean(full_weights))))
    return dict(scenes=records, conditions={key: {metric: float(np.mean([r['conditions'][key][metric] for r in records]))
        for metric in ['iou', 'class_accuracy', 'instance_accuracy']} for key in records[0]['conditions']},
        errors={key: {kind: sum(r['conditions'][key]['errors'][kind] for r in records)
            for kind in records[0]['conditions'][key]['errors']} for key in records[0]['conditions']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['semantic', 'frozen', 'joint'], required=True)
    args = p.parse_args();torch.set_num_threads(3);torch.manual_seed(20261006);np.random.seed(20261006)
    ROOT.mkdir(exist_ok=True);design = json.loads((SOURCE / 'design.json').read_text())
    if args.stage == 'semantic':
        write(ROOT / 'design.json', dict(fit=design['fit'], calibration=design['calibration'], validation=design['validation'],
            source=str(SOURCE), mechanisms=['semantic channel mixture', 'independent conditional relation'],
            fits=['semantic', 'frozen', 'joint'], epochs_per_fit=12, seed=20261006,
            supervision='independent NR3D, ScanEnts3D roles, official3D instance/category labels',
            unknown_policy='semantic normalization weighted by annotated region fraction; unknown is not negative',
            semantic_selection='calibration scene-macro semantic-class accuracy',
            relation_selection='calibration actual selected-set IoU; zero relation checkpoint included',
            gate=dict(minimum_iou_gain_over_previous=.02, minimum_iou_gain_over_semantic=.02,
                must_beat_joint=True, no_class_accuracy_drop_from_semantic=True, minimum_scene_evidence_gain=.005),
            parameter_search=False, source_images_at_query=False, independent_final_test=False))
    groups = {key: [load_scene(scene) for scene in design[key]] for key in ['fit', 'calibration', 'validation']}
    for scenes in groups.values():
        for data in scenes:semantic_targets(data)
    architecture = dict(text_width=1152, width=64, native_scale=native_logit_scale(ENCODER))
    model = TypedReferentBinding(**architecture).cuda()
    if args.stage != 'semantic':
        checkpoint = torch.load(ROOT / 'semantic/model.pt', map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
    model.identity.requires_grad_(args.stage != 'frozen')
    model.relation.requires_grad_(args.stage != 'semantic')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=.01)
    rng = np.random.default_rng(20261006);out = ROOT / args.stage;out.mkdir(exist_ok=True)
    eligible = [];counts = []
    for data in groups['fit']:
        listener = [j for j, row in enumerate(data['rows']) if row['listener_correct'] and float(data['reward'][j].max()) > 0]
        eligible.append(np.array([j for j in listener if data['rows'][j]['mentions_target_class']]))
        counts.append(len(listener))
    def save(epoch):
        torch.save(dict(model_type='typed_referent_binding', architecture=architecture, stage=args.stage,
            state_dict={k: v.cpu() for k, v in model.state_dict().items()}, epoch=epoch,
            role_checkpoint=str((SOURCE / 'roles/model.pt').resolve()), training_design=str(ROOT / 'design.json')),
            out / 'model.pt')
    measure = 'class_accuracy' if args.stage == 'semantic' else 'iou'
    start = time.monotonic();initial = evaluate(model, groups['calibration'])
    best = initial['conditions']['original'][measure];save(0);history = [dict(epoch=0, calibration=initial)]
    for epoch in range(1, 13):
        tasks = []
        for i, ids in enumerate(eligible):
            shuffled = ids.copy();rng.shuffle(shuffled);shuffled = np.resize(shuffled, counts[i])
            tasks.extend((i, shuffled[j:j + 8]) for j in range(0, len(shuffled), 8))
        rng.shuffle(tasks);model.train();losses = []
        for scene_index, ids in tasks:
            data = groups['fit'][scene_index];optimizer.zero_grad(set_to_none=True);accumulated = 0.
            active = 0
            for index in ids:
                query = data['queries'][index]
                if args.stage == 'frozen' and not bool((query['roles'] == 2).any()):
                    # No relation is invoked; freezing semantic identity means
                    # this example has no trainable output. It remains evaluated.
                    continue
                with torch.autocast('cuda', dtype=torch.bfloat16):scores = model(query)
                if args.stage == 'semantic':
                    target = data['class_target'][index]
                    observed = data['known_fraction'] > 0
                    log_probability = F.log_softmax(scores[observed].float() + data['known_fraction'][observed].log(), dim=-1)
                    loss = -(target[observed] * log_probability).sum() / len(ids)
                else:
                    target = data['reward'][index].pow(4);target = target / target.sum()
                    loss = -(target * F.log_softmax(scores.float(), dim=-1)).sum() / len(ids)
                loss.backward();accumulated += float(loss);active += 1
            if active:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.);optimizer.step()
            losses.append(accumulated)
        calibration = evaluate(model, groups['calibration']);value = calibration['conditions']['original'][measure]
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), calibration=calibration))
        if value > best:best = value;save(epoch)
        print(args.stage, epoch, 'loss', round(np.mean(losses), 4), 'calibration', round(value, 5), 'best', round(best, 5), flush=True)
    saved = torch.load(out / 'model.pt', map_location='cuda', weights_only=False);model.load_state_dict(saved['state_dict'])
    report = dict(stage=args.stage, architecture=architecture, selected_epoch=saved['epoch'], history=history,
        calibration=evaluate(model, groups['calibration'], True), validation=evaluate(model, groups['validation'], True),
        seconds=time.monotonic() - start, eligible_queries=sum(map(len, eligible)), presentations_per_epoch=sum(counts),
        total_parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        benchmark_data_used=False, native_coordinates_changed=False, scope_changed=False,
        semantic_unknown_regions_used_as_negatives=False)
    if args.stage != 'semantic':
        reference = torch.load(ROOT / 'semantic/model.pt', map_location='cpu', weights_only=False)['state_dict']
        report['semantic_parameters_unchanged'] = all(torch.equal(value.cpu(), reference[key])
            for key, value in model.state_dict().items() if key.startswith('identity.'))
    write(out / 'fit.json', report);print('complete', args.stage, report['validation']['conditions'], flush=True)


if __name__ == '__main__':main()
