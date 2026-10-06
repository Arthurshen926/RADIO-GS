"""Separate extraction, teacher recognition and conditional spatial use."""
import json

import numpy as np
import torch
from torch.nn import functional as F

from scripts.research.fit_grounded_scene_binding import load_scene
from scripts.research.prepare_native_phrase_binding import ROOT, write
from scripts.research.prepare_referent_grounding import ENCODER
from radio_gs.querying.referent_identity import native_logit_scale


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    design = json.loads((ROOT / 'design.json').read_text());groups = {}
    for split in ['calibration', 'validation']:
        records = []
        for scene in design[split]:
            data = load_scene(scene, 'cuda')
            encoded = np.load(ROOT / 'phrases' / (scene + '.npz'))
            phrases = json.loads((ROOT / 'phrases' / (scene + '.json')).read_text())
            vectors = F.normalize(torch.from_numpy(encoded['pooled'].astype(np.float32)).cuda(), dim=-1)
            scores = vectors @ data['visual'].T
            rows = []
            for qi, predicted in enumerate(phrases):
                truth = data['entities'][qi]
                for pi, phrase in enumerate(predicted):
                    overlap = [len(set(phrase['tokens']) & set(entity['tokens'])) /
                        len(set(phrase['tokens']) | set(entity['tokens'])) for entity in truth]
                    match = truth[int(np.argmax(overlap))] if overlap and max(overlap) > .5 else None
                    row = dict(query=qi, phrase=phrase['phrase'], role=phrase['role'], matched=match is not None)
                    if match is not None:
                        score = scores[int(encoded['offsets'][qi]) + pi]
                        mass = (score * native_logit_scale(ENCODER)).softmax(-1)
                        seed = int(score.argmax());purity = data['entity_purities'][match['columns']].sum(0)
                        row.update(role_correct=phrase['role'] == (1 if match['target'] else 2),
                            native_majority=bool(purity[seed] > .5),
                            native_scope_iou=float(data['entity_rewards'][match['columns'], seed].max()),
                            posterior_correct_majority_mass=float(mass[purity > .5].sum()),
                            landmark=not match['target'])
                    rows.append(row)
            matched = [r for r in rows if r['matched']]
            metrics = dict(predicted=len(rows), matched=len(matched),
                matched_fraction=len(matched) / max(len(rows), 1),
                **{key: float(np.mean([r[key] for r in matched])) for key in
                    ['role_correct', 'native_majority', 'native_scope_iou', 'posterior_correct_majority_mass']})
            records.append(dict(scene=scene, metrics=metrics, rows=rows))
            print(split, scene, metrics, flush=True)
        groups[split] = dict(scenes=records, scene_macro={key: float(np.mean([r['metrics'][key] for r in records]))
            for key in ['matched_fraction', 'role_correct', 'native_majority', 'native_scope_iou', 'posterior_correct_majority_mass']})
    write(ROOT / 'phrase_diagnosis.json', dict(groups=groups,
        source='predicted query-only phrases; annotations used only to evaluate extraction/recognition',
        phrase_match_rule='subword-set IoU greater than0.5; no filtering at inference', benchmark_labels_used=False))
    fits = {key: json.loads((ROOT / key / 'fit.json').read_text()) for key in ['native', 'uniform']}
    candidate = fits['native']['validation']['conditions'];control = fits['uniform']['validation']['conditions']
    previous = json.loads(open('output/real_referent_grounding_20261005/local_scope/fit.json').read())
    previous_iou = previous['validation']['scene_macro_iou']
    gate = dict(candidate=candidate['original']['iou'], control=control['original']['iou'], previous_local=previous_iou,
        uniform_intervention=candidate['uniform']['iou'], query_mean=candidate['query_mean']['iou'],
        reversed_witness=candidate['reversed']['iou'],
        passed=(candidate['original']['iou'] > max(control['original']['iou'], previous_iou) and
            candidate['original']['iou'] >= candidate['uniform']['iou'] + .005),
        benchmark_metrics_used=False, default='system_v9',
        explanation='One fixed search round; protect native phrase recognition and require actual conditional spatial utility.')
    write(ROOT / 'development_gate.json', gate)
    write(ROOT / 'comparison.json', dict(development_gate=gate,
        fits={key: {k: value[k] for k in ['selected_epoch', 'parameter_count', 'seconds', 'validation', 'calibration']}
              for key, value in fits.items()}, phrase_diagnosis=groups,
        role_extraction=json.loads((ROOT / 'roles/fit.json').read_text())['groups']))
    print('gate', gate, flush=True)


if __name__ == '__main__':
    main()
