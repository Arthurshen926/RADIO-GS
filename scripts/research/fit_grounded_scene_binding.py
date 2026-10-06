"""Matched scene binding fits with/without explicit phrase-instance supervision."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.querying.grounded_scene_binding import GroundedSceneBinding
from radio_gs.querying.referent_identity import native_logit_scale
from scripts.research.fit_referent_grounder import read_scene, evaluate
from scripts.research.prepare_grounded_scene_binding import ROOT
from scripts.research.prepare_referent_grounding import ENCODER


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_scene(scene, device):
    data = read_scene(scene, device, ROOT)
    annotations = json.loads((ROOT / 'entities' / (scene + '.json')).read_text())
    targets = np.load(ROOT / 'entities' / (scene + '.npz'))
    rewards = targets['reward'][data['ids']].T
    normalized = rewards ** 4
    normalized /= np.maximum(normalized.sum(1, keepdims=True), 1e-30)
    data['entity_distributions'] = torch.from_numpy(normalized).to(device)
    data['entity_rewards'] = torch.from_numpy(rewards).to(device)
    data['entity_purities'] = torch.from_numpy(targets['purity'][data['ids']].T).to(device)
    data['entities'] = annotations
    return data


def entity_loss(logits, annotations, distributions):
    terms = []
    for mention in annotations:
        # Score one phrase once, even when the tokenizer splits its words.
        scores = logits[mention['tokens']].float().mean(0)
        target = distributions[mention['columns']].mean(0)
        terms.append(-(target * F.log_softmax(scores, dim=-1)).sum())
    return torch.stack(terms).mean() if terms else logits.sum() * 0


@torch.inference_mode()
def diagnose(model, scenes):
    model.eval();records = []
    for data in scenes:
        scores = [];no_witness = [];entities = []
        native = data['pooled'] @ data['visual'].T * model.native_scale
        for index in range(len(data['rows'])):
            sl = slice(index, index + 1)
            args = (data['visual'], data['geometry'], data['tokens'][sl], data['valid'][sl], data['pooled'][sl])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result, mentions = model(*args, return_mentions=True)
                uniform = model(*args, grounded_context=False)
            scores.append(result[0].float());no_witness.append(uniform[0].float())
            for mention in data['entities'][index]:
                seed = int(mentions[0, mention['tokens']].float().mean(0).argmax())
                columns = mention['columns']
                entities.append(dict(target=mention['target'],
                    iou=float(data['entity_rewards'][columns, seed].max()),
                    majority=float(data['entity_purities'][columns, seed].sum()) > .5))
        learned = torch.stack(scores);delta = learned - native
        centred = delta - delta.mean(1, keepdim=True);common = centred.mean(0, keepdim=True)
        row = torch.arange(len(scores), device=native.device)
        conditions = {}
        for key, logits in dict(native=native, original=learned, query_mean=native + common,
                                uniform_witness=torch.stack(no_witness)).items():
            seed = logits.argmax(1)
            conditions[key] = dict(iou=float(data['reward'][row, seed].mean()),
                majority=float((data['purity'][row, seed] > .5).float().mean()),
                same_seed=float((seed == learned.argmax(1)).float().mean()),
                per_query_iou=data['reward'][row, seed].cpu().tolist())
        mention_metrics = {}
        for kind in ['all', 'target', 'landmark']:
            rows = [r for r in entities if kind == 'all' or r['target'] == (kind == 'target')]
            mention_metrics[kind] = dict(count=len(rows), iou=float(np.mean([r['iou'] for r in rows])) if rows else None,
                majority=float(np.mean([r['majority'] for r in rows])) if rows else None)
        records.append(dict(scene=data['scene'], conditions=conditions, mentions=mention_metrics,
            query_common_energy=float(common.square().mean() / centred.square().mean().clamp_min(1e-12))))
    return dict(scenes=records, conditions={k: {metric: float(np.mean([r['conditions'][k][metric] for r in records]))
        for metric in ['iou', 'majority', 'same_seed']} for k in ['native', 'original', 'query_mean', 'uniform_witness']},
        mentions={kind: {metric: float(np.mean([r['mentions'][kind][metric] for r in records
            if r['mentions'][kind][metric] is not None])) for metric in ['iou', 'majority']} for kind in ['all', 'target', 'landmark']},
        query_common_energy=float(np.mean([r['query_common_energy'] for r in records])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=['ungrounded', 'grounded'], required=True)
    args = parser.parse_args();torch.set_num_threads(3)
    torch.manual_seed(20261005);np.random.seed(20261005)
    design = json.loads((ROOT / 'design.json').read_text())
    groups = {k: [load_scene(s, 'cuda') for s in design[k]] for k in ['fit', 'calibration', 'validation']}
    architecture = dict(width=128, text_width=1152, native_width=1536, native_scale=native_logit_scale(ENCODER))
    model = GroundedSceneBinding(**architecture).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rng = np.random.default_rng(20261005);out = ROOT / args.condition;out.mkdir(exist_ok=True)
    eligible = [];counts = [];entity_count = 0
    for data in groups['fit']:
        listener = [j for j, q in enumerate(data['rows']) if q['listener_correct'] and float(data['reward'][j].max()) > 0]
        ids = np.array([j for j in listener if data['rows'][j]['mentions_target_class']])
        eligible.append(ids);counts.append(len(listener));entity_count += sum(len(data['entities'][i]) for i in ids)
    def save(epoch):
        torch.save(dict(model_type='grounded_scene_binding', architecture=architecture,
            state_dict={k: v.cpu() for k, v in model.state_dict().items()}, epoch=epoch,
            training_condition='disjoint real RGB-D rooms; NR3D and official3D instances' +
                (' plus ScanEnts3D phrase-instance grounding' if args.condition == 'grounded' else ''),
            fit=design['fit'], calibration=design['calibration'], condition=args.condition,
            target='scope IoU^4; optional explicit scene mention alignment'), out / 'model.pt')
    start = time.monotonic();initial = evaluate(model, groups['calibration'], 1)
    best = initial['scene_macro_iou'];save(0);history = [dict(epoch=0, calibration=initial)]
    for epoch in range(1, 13):
        tasks = []
        for i, ids in enumerate(eligible):
            shuffled = ids.copy();rng.shuffle(shuffled);shuffled = np.resize(shuffled, counts[i])
            tasks.extend((i, shuffled[j:j + 8]) for j in range(0, len(shuffled), 8))
        rng.shuffle(tasks);model.train();losses = [];alignment = []
        for scene_index, ids in tasks:
            data = groups['fit'][scene_index];optimizer.zero_grad(set_to_none=True)
            accumulated = 0.;entity_accumulated = 0.
            for index in ids:
                sl = slice(index, index + 1)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    scores, mentions = model(data['visual'], data['geometry'], data['tokens'][sl],
                                             data['valid'][sl], data['pooled'][sl], return_mentions=True)
                target = data['reward'][index].pow(4);target = target / target.sum()
                main_loss = -(target * F.log_softmax(scores[0].float(), dim=-1)).sum()
                auxiliary = entity_loss(mentions[0], data['entities'][index], data['entity_distributions'])
                loss = (main_loss + auxiliary if args.condition == 'grounded' else main_loss) / len(ids)
                loss.backward();accumulated += float(main_loss) / len(ids);entity_accumulated += float(auxiliary) / len(ids)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.);optimizer.step()
            losses.append(accumulated);alignment.append(entity_accumulated)
        calibration = evaluate(model, groups['calibration'], 1);value = calibration['scene_macro_iou']
        history.append(dict(epoch=epoch, referent_loss=float(np.mean(losses)),
            mention_loss=float(np.mean(alignment)), calibration=calibration))
        if value > best:
            best = value;save(epoch)
        print(args.condition, epoch, 'main', round(np.mean(losses), 4), 'mention', round(np.mean(alignment), 4),
              'calibration', round(value, 5), 'best', round(best, 5), flush=True)
    saved = torch.load(out / 'model.pt', map_location='cuda', weights_only=False);model.load_state_dict(saved['state_dict'])
    report = dict(condition=args.condition, architecture=architecture, selected_epoch=saved['epoch'], epochs=12,
        eligible_records=sum(map(len, eligible)), eligible_mentions=entity_count, presentations_per_epoch=sum(counts),
        parameter_count=sum(p.numel() for p in model.parameters()), history=history,
        calibration=evaluate(model, groups['calibration'], 1), validation=evaluate(model, groups['validation'], 1),
        seconds=time.monotonic() - start, native_coordinates_changed=False, scope_changed=False, benchmark_data_used=False)
    write(out / 'fit.json', report)
    write(out / 'diagnosis.json', {k: diagnose(model, groups[k]) for k in ['calibration', 'validation']})
    print('complete', args.condition, report['validation']['scene_macro_iou'], flush=True)


if __name__ == '__main__':
    main()
