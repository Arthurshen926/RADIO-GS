"""Privileged independent-room interventions, never deployment predictions.

Keep the fitted decoder and native map fixed. Separately supply annotated phrase
roles and confirmed landmark identity to locate remaining upstream losses.
"""
import json
import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.models.native_query_encoders import load_siglip_text
from radio_gs.querying.native_phrase_binding import NativePhraseBinding, spatial_witnesses
from scripts.research.fit_grounded_scene_binding import load_scene
from scripts.research.prepare_native_phrase_binding import ROOT, write
from scripts.research.prepare_referent_grounding import ENCODER


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    saved = torch.load(ROOT / 'native/model.pt', map_location='cuda', weights_only=False)
    model = NativePhraseBinding(**saved['architecture']).cuda().eval();model.load_state_dict(saved['state_dict'])
    teacher, tokenizer = load_siglip_text(ENCODER, 'cuda')
    design = json.loads((ROOT / 'design.json').read_text());records = []
    for scene in design['validation']:
        data = load_scene(scene, 'cuda');phrases = [e['phrase'] for row in data['entities'] for e in row]
        features = []
        for start in range(0, len(phrases), 24):
            batch = tokenizer(phrases[start:start + 24], padding='max_length', max_length=64,
                truncation=True, return_tensors='pt').to('cuda')
            features.append(F.normalize(F.normalize(teacher(**batch).pooler_output.float(), dim=-1).half().float(), dim=-1))
        scores = torch.cat(features) @ data['visual'].T
        native_spatial = spatial_witnesses(data['geometry'], scores, model.native_scale)
        masses = []
        for annotations in data['entities']:
            for entity in annotations:
                # Same additional3D labels as previous training diagnostics.
                # Equal instance mass; scopeIoU^4 downweights impure supports.
                mass = data['entity_rewards'][entity['columns']].pow(4)
                mass = mass / mass.sum(1, keepdim=True)
                masses.append(mass.mean(0))
        gold_spatial = spatial_witnesses(data['geometry'], torch.stack(masses).clamp_min(1e-30).log() / model.native_scale,
            model.native_scale)['native']
        predicted_cache = np.load(ROOT / 'phrases' / (scene + '.npz'))
        predicted_rows = json.loads((ROOT / 'phrases' / (scene + '.json')).read_text())
        predicted_scores = F.normalize(torch.from_numpy(predicted_cache['pooled'].astype(np.float32)).cuda(), dim=-1) @ data['visual'].T
        predicted_spatial = spatial_witnesses(data['geometry'], predicted_scores, model.native_scale)
        native = data['pooled'] @ data['visual'].T
        values = {key: [] for key in ['predicted', 'annotated_phrases', 'annotated_phrases_and_landmarks', 'annotated_phrases_uniform']}
        unannotated = 0;offset = 0;rows = []
        for i, annotations in enumerate(data['entities']):
            pred = predicted_rows[i];sl = slice(predicted_cache['offsets'][i], predicted_cache['offsets'][i + 1])
            base = dict(geometry=data['geometry'], native=native[i], query_tokens=data['tokens'][i, data['valid'][i]].mean(0))
            pred_args = dict(base, phrase_scores=predicted_scores[sl],
                phrase_tokens=torch.stack([data['tokens'][i, e['tokens']].mean(0) for e in pred]) if pred else data['tokens'].new_empty((0, 1152)),
                roles=torch.tensor([e['role'] for e in pred], device='cuda'),
                spatial={key: value[sl] for key, value in predicted_spatial.items()})
            with torch.autocast('cuda', dtype=torch.bfloat16):
                guesses = {'predicted': model(**pred_args)}
                if annotations:
                    sl = slice(offset, offset + len(annotations))
                    args = dict(base, phrase_scores=scores[sl],
                        phrase_tokens=torch.stack([data['tokens'][i, e['tokens']].mean(0) for e in annotations]),
                        roles=torch.tensor([1 if e['target'] else 2 for e in annotations], device='cuda'),
                        spatial={key: value[sl] for key, value in native_spatial.items()})
                    guesses['annotated_phrases'] = model(**args)
                    guesses['annotated_phrases_uniform'] = model(**args, witness='uniform')
                    args['spatial'] = {'native': gold_spatial[sl]}
                    guesses['annotated_phrases_and_landmarks'] = model(**args)
                else:
                    unannotated += 1
                    guesses.update({key: guesses['predicted'] for key in values if key != 'predicted'})
            offset += len(annotations);row = dict(query=i, expression=data['rows'][i]['expression'], annotated=bool(annotations), conditions={})
            for key, logits in guesses.items():
                seed = int(logits.argmax());iou = float(data['reward'][i, seed]);majority = float(data['purity'][i, seed]) > .5
                values[key].append((iou, majority));row['conditions'][key] = dict(iou=iou, majority=majority)
            rows.append(row)
        metrics = {key: dict(iou=float(np.mean(v, axis=0)[0]), majority=float(np.mean(v, axis=0)[1])) for key, v in values.items()}
        records.append(dict(scene=scene, queries=len(rows), unannotated_fallback=unannotated, conditions=metrics, rows=rows))
        print(scene, metrics, flush=True)
    write(ROOT / 'privileged_interventions.json', dict(diagnostic_only=True, model_frozen=True, map_frozen=True,
        annotations_used='independent development ScanEnts3D phrase roles and official3D landmark identity',
        automatic_method_result=False, system_upper_bound=False,
        scenes=records, conditions={key: {metric: float(np.mean([r['conditions'][key][metric] for r in records]))
            for metric in ['iou', 'majority']} for key in records[0]['conditions']}))


if __name__ == '__main__':
    main()
