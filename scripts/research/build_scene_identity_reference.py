"""Query-native reference population not conditioned on detector admission.

All declared source-region descriptors participate. One-vs-two density evidence
uses the same model class as current identity, but its reference includes regions
the detector never proposed. It is not a calibrated foreground probability.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.mixture import GaussianMixture


def fit_reference(scores, candidate_scores):
    x = np.asarray(scores, dtype=np.float64)
    center, scale = x.mean(), max(x.std(), 1e-6)
    z = ((x-center)/scale)[:, None]
    one = GaussianMixture(1, random_state=0).fit(z)
    two = GaussianMixture(2, n_init=5, random_state=0).fit(z)
    probability = 1/(1+np.exp(np.clip((two.bic(z)-one.bic(z))/2, -700, 700)))
    high = int(two.means_[:, 0].argmax())
    query = ((np.asarray(candidate_scores)-center)/scale)[:, None]
    posterior = (1-probability)+probability*two.predict_proba(query)[:, high] if len(query) else np.empty(0)
    return posterior, dict(samples=len(x), bic_one=float(one.bic(z)), bic_two=float(two.bic(z)),
        two_probability=float(probability), means=(two.means_[:, 0]*scale+center).tolist(), weights=two.weights_.tolist())


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--identity-record', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    start = time.monotonic()
    inputs = json.loads(Path(a.inputs).read_text())
    scene = json.loads(Path(inputs['scene']).read_text())
    response = torch.load(inputs['response'], weights_only=False, map_location='cpu')
    q = F.normalize(response['encodings'].float(), dim=-1)
    identity = json.loads(Path(a.identity_record).read_text())
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    banks, frame_ids = [], []
    for frame in scene['source_frames']:
        for directory in scene['native_region_directories']:
            native = torch.load(Path(directory)/f'frame_{frame:05d}.pt', weights_only=False, map_location='cpu')
            scores = F.normalize(native['descriptors'].float(), dim=-1) @ q.T
            banks.append(scores)
            frame_ids.extend([frame]*len(scores))
    bank = torch.cat(banks)
    torch.save(dict(scores=bank, frames=torch.tensor(frame_ids), texts=response['texts']), out/'reference.pt')
    models, matches = [], {}
    for qi, text in enumerate(response['texts']):
        entries = [(frame, mi, row['native_score']) for frame, rows in identity['matches'].items()
            for mi, row in enumerate(rows[qi])
            if (row['identity_score_valid'] if 'identity_score_valid' in row else row['region_iou'] >= identity['minimum_region_iou'])]
        posterior, model = fit_reference(bank[:, qi].numpy(), [score for _, _, score in entries])
        models.append(dict(query=text, **model))
        for (frame, mi, score), prob in zip(entries, posterior):
            matches[f'{frame}:{qi}:{mi}'] = dict(native_score=score, reference_support=float(prob))
    (out/'reference.json').write_text(json.dumps(dict(inputs=a.inputs, identity_record=a.identity_record,
        source_frames=scene['source_frames'], region_count=len(bank), queries=response['texts'], models=models,
        matches=matches, new_vfm_calls=0, posterior_is_calibrated_correctness=False,
        source_rgb_or_gt_read=False, seconds=time.monotonic()-start), indent=2)+'\n')
    print(scene['scene'], len(bank), 'source region references complete', flush=True)


if __name__ == '__main__':
    main()
