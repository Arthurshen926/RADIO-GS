"""Learn query-only mention roles, then encode predicted phrases with SigLIP.

Human phrase annotations supervise only the extractor in independent training
rooms. The encoded phrases in every split are its predictions, not annotations.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.models.native_query_encoders import load_siglip_text
from radio_gs.querying.native_phrase_binding import MentionRoles, decode_mentions
from scripts.research.prepare_grounded_scene_binding import ROOT as SOURCE
from scripts.research.prepare_referent_grounding import ENCODER

ROOT = Path('output/native_phrase_binding_20261006')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def span_set(labels):
    spans = [];active = None
    for i, tag in enumerate(labels):
        role = 1 if tag in (1, 2) else 2 if tag in (3, 4) else 0
        if not role:
            active = None
        elif active is None or active[0] != role or tag in (1, 3):
            active = [role, i, i + 1];spans.append(active)
        else:
            active[2] = i + 1
    return set(map(tuple, spans))


@torch.inference_mode()
def evaluate(model, data):
    model.eval();correct = predicted = actual = tokens = token_correct = 0
    for scene in data:
        logits = model(scene['tokens'])
        tags = logits.argmax(-1).cpu().numpy()
        for i in scene['annotated']:
            valid = scene['valid'][i].cpu().numpy()
            guess = tags[i].copy();guess[~valid] = 0
            truth = scene['labels'][i].cpu().numpy();truth = np.maximum(truth, 0)
            a, b = span_set(guess), span_set(truth)
            correct += len(a & b);predicted += len(a);actual += len(b)
            tokens += int(valid.sum());token_correct += int((guess[valid] == truth[valid]).sum())
    return dict(exact_token_span_f1=2 * correct / max(predicted + actual, 1),
        precision=correct / max(predicted, 1), recall=correct / max(actual, 1),
        token_accuracy=token_correct / max(tokens, 1), correct=correct, predicted=predicted, annotated=actual)


def main():
    torch.set_num_threads(3);torch.manual_seed(20261006);np.random.seed(20261006)
    ROOT.mkdir(exist_ok=True)
    design = json.loads((SOURCE / 'design.json').read_text())
    for name in ['maps', 'language', 'labels', 'entities']:
        if not (ROOT / name).exists():
            (ROOT / name).symlink_to((SOURCE / name).resolve(), target_is_directory=True)
    write(ROOT / 'design.json', {**design, 'mechanism': 'native pooled phrase witnesses with query-only learned roles',
        'conditions': ['uniform', 'native'], 'role_epochs': 18, 'binding_epochs': 12,
        'source': str(SOURCE), 'gate': 'native beats uniform and previous local9.7857% on four development rooms; native witnesses beat uniform intervention by0.5pp',
        'benchmark_inference_phrase_annotations': False, 'search_rounds': 2,
        'spatial_basis': 'all-region marginalization; room-normalized Gaussian proximity at0.05/0.15/0.5 and six signed axis responses at0.05',
        'round2': dict(declared_after='round1 privileged interventions: correct phrases/landmarks fail to repair frozen relation use',
            conditions=['native_instance', 'uniform_instance'], epochs=12,
            loss='scopeIoU CE plus equal-weight same-class instance CE on relation output only; region-purity^4 normalized per instance',
            selection='same three calibration rooms',
            gate='exceeds native round1 and matched uniform_instance; uniform intervention costs at least0.5pp; query mean inferior',
            source_labels='same independent official3D labels, now using their semantic classes', further_parameter_scans=False)})
    groups = {}
    for split in ['fit', 'calibration', 'validation']:
        groups[split] = []
        for scene in design[split]:
            cache = np.load(ROOT / 'language' / (scene + '.npz'))
            rows = json.loads((ROOT / 'language' / (scene + '.json')).read_text())
            annotations = json.loads((ROOT / 'entities' / (scene + '.json')).read_text())
            labels = np.full(cache['valid'].shape, -100, np.int64);annotated = []
            for i, mentions in enumerate(annotations):
                eligible = split != 'fit' or (rows[i]['listener_correct'] and rows[i]['mentions_target_class'])
                if not mentions or not eligible:
                    continue
                annotated.append(i);labels[i, cache['valid'][i]] = 0
                for mention in mentions:
                    ids = mention['tokens'];begin = 1 if mention['target'] else 3
                    labels[i, ids] = begin + 1;labels[i, ids[0]] = begin
            groups[split].append(dict(scene=scene, rows=rows, annotations=annotations, annotated=annotated,
                tokens=torch.from_numpy(cache['tokens'].astype(np.float32)).cuda(),
                valid=torch.from_numpy(cache['valid']).cuda(), labels=torch.from_numpy(labels).cuda()))
    model = MentionRoles().cuda();opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    x = torch.cat([d['tokens'][d['annotated']] for d in groups['fit']])
    y = torch.cat([d['labels'][d['annotated']] for d in groups['fit']])
    counts = torch.bincount(y[y >= 0], minlength=5).float()
    weights = counts.sum().sqrt() / counts.sqrt();weights /= weights.mean()
    history = [];best = -1;start = time.monotonic();out = ROOT / 'roles';out.mkdir(exist_ok=True)
    for epoch in range(1, 19):
        model.train();order = torch.randperm(len(x), device='cuda');losses = []
        for offset in range(0, len(order), 32):
            ids = order[offset:offset + 32];opt.zero_grad(set_to_none=True)
            logits = model(x[ids]);loss = F.cross_entropy(logits.flatten(0, 1), y[ids].flatten(), weight=weights)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(), 1.);opt.step();losses.append(float(loss))
        metrics = evaluate(model, groups['calibration'])
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), calibration=metrics))
        if metrics['exact_token_span_f1'] > best:
            best = metrics['exact_token_span_f1']
            torch.save(dict(state_dict=model.state_dict(), architecture={}, epoch=epoch), out / 'model.pt')
        print('roles', epoch, metrics, flush=True)
    saved = torch.load(out / 'model.pt', weights_only=False);model.load_state_dict(saved['state_dict']);model.eval()
    write(out / 'fit.json', dict(selected_epoch=saved['epoch'], fit_queries=len(x), class_counts=counts.cpu().tolist(),
        history=history, groups={key: evaluate(model, data) for key, data in groups.items()}, seconds=time.monotonic() - start,
        additional_supervision='independent ScanEnts3D target/landmark phrase spans', benchmark_labels_used=False))
    del x, y
    teacher, tokenizer = load_siglip_text(ENCODER, 'cuda')
    folder = ROOT / 'phrases';folder.mkdir(exist_ok=True);records = []
    with torch.inference_mode():
        for split, scenes in groups.items():
            for data in scenes:
                expressions = [row['expression'] for row in data['rows']]
                offsets = tokenizer(expressions, padding='max_length', max_length=64, truncation=True,
                    return_offsets_mapping=True)['offset_mapping']
                logits = model(data['tokens']).cpu().numpy()
                mentions = [decode_mentions(q, o, l) for q, o, l in zip(expressions, offsets, logits)]
                phrases = [m['phrase'] for row in mentions for m in row]
                pooled = []
                for start in range(0, len(phrases), 24):
                    batch = tokenizer(phrases[start:start + 24], padding='max_length', max_length=64,
                        truncation=True, return_tensors='pt').to('cuda')
                    pooled.append(F.normalize(teacher(**batch).pooler_output.float(), dim=-1).half().cpu().numpy())
                np.savez_compressed(folder / (data['scene'] + '.npz'),
                    pooled=np.concatenate(pooled) if pooled else np.empty((0, 1536), np.float16),
                    offsets=np.cumsum([0] + [len(row) for row in mentions]))
                write(folder / (data['scene'] + '.json'), mentions)
                record = dict(scene=data['scene'], split=split, queries=len(mentions), phrases=len(phrases),
                    no_target=sum(not any(m['role'] == 1 for m in row) for row in mentions),
                    no_landmark=sum(not any(m['role'] == 2 for m in row) for row in mentions))
                records.append(record);print('encoded', record, flush=True)
    write(ROOT / 'preparation.json', dict(scenes=records, phrase_source='query-only frozen role extractor',
        annotations_in_encoded_phrase_selection=False, native_teacher_frozen=True))


if __name__ == '__main__':
    main()
