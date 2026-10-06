"""Privileged phrase-recognition diagnostic, never an automatic task score.

Published phrase boundaries are supplied only here to separate native teacher
recognition from learned mention binding. They are unavailable to deployment.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.models.native_query_encoders import load_siglip_text
from radio_gs.querying.grounded_scene_binding import GroundedSceneBinding
from scripts.research.fit_grounded_scene_binding import load_scene, write
from scripts.research.prepare_grounded_scene_binding import ROOT
from scripts.research.prepare_referent_grounding import ENCODER


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    teacher, tokenizer = load_siglip_text(ENCODER, 'cuda')
    design = json.loads((ROOT / 'design.json').read_text())
    report = {}
    models = {}
    for condition in ['ungrounded', 'grounded']:
        saved = torch.load(ROOT / condition / 'model.pt', map_location='cuda', weights_only=False)
        model = GroundedSceneBinding(**saved['architecture']).cuda().eval();model.load_state_dict(saved['state_dict'])
        models[condition] = model
    for split in ['calibration', 'validation']:
        scenes = []
        for scene in design[split]:
            data = load_scene(scene, 'cuda')
            phrases = list(dict.fromkeys(e['phrase'] for row in data['entities'] for e in row))
            encoded = []
            for start in range(0, len(phrases), 16):
                batch = tokenizer(phrases[start:start + 16], padding='max_length', max_length=64,
                    truncation=True, return_tensors='pt').to('cuda')
                encoded.append(F.normalize(teacher(**batch).pooler_output.float(), dim=-1))
            native = torch.cat(encoded) @ data['visual'].T
            phrase_indices = {p: i for i, p in enumerate(phrases)};rows = []
            for qi, annotations in enumerate(data['entities']):
                head = teacher.text_model.head
                projected = F.linear(data['tokens'][qi].float(), head.weight.float(), head.bias.float())
                native_tokens = F.normalize(projected, dim=-1) @ data['visual'].T
                learned = {}
                for condition, model in models.items():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        _, logits = model(data['visual'], data['geometry'], data['tokens'][qi:qi + 1],
                            data['valid'][qi:qi + 1], data['pooled'][qi:qi + 1], return_mentions=True)
                    learned[condition] = logits[0].float()
                for entity in annotations:
                    seeds = dict(native_phrase=int(native[phrase_indices[entity['phrase']]].argmax()),
                        native_contextual_tokens=int(native_tokens[entity['tokens']].mean(0).argmax()))
                    seeds.update({c: int(score[entity['tokens']].mean(0).argmax()) for c, score in learned.items()})
                    metrics = {c: dict(iou=float(data['entity_rewards'][entity['columns'], seed].max()),
                        majority=bool(data['entity_purities'][entity['columns'], seed].sum() > .5)) for c, seed in seeds.items()}
                    rows.append(dict(query=qi, phrase=entity['phrase'], target=entity['target'], metrics=metrics))
            aggregate = {}
            for kind in ['all', 'target', 'landmark']:
                selected = [r for r in rows if kind == 'all' or r['target'] == (kind == 'target')]
                aggregate[kind] = {c: {metric: float(np.mean([r['metrics'][c][metric] for r in selected]))
                    for metric in ['iou', 'majority']} for c in ['native_phrase', 'native_contextual_tokens', 'ungrounded', 'grounded']}
            scenes.append(dict(scene=scene, rows=rows, aggregate=aggregate))
            print(split, scene, aggregate, flush=True)
        report[split] = dict(scenes=scenes, aggregate={kind: {c: {metric: float(np.mean([
            s['aggregate'][kind][c][metric] for s in scenes])) for metric in ['iou', 'majority']}
            for c in ['native_phrase', 'native_contextual_tokens', 'ungrounded', 'grounded']} for kind in ['all', 'target', 'landmark']})
    write(ROOT / 'native_phrase_diagnosis.json', dict(diagnostic_only=True,
        privileged_input='published annotated phrase boundary and wording; not supplied at deployed query time',
        same_frozen_maps_and_scope=True, groups=report))


if __name__ == '__main__':
    main()
