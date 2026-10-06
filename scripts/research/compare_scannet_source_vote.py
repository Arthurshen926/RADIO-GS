"""Same-evidence simple backprojection baseline for the current scope solver.

No score from the fitted solver is consumed. The baseline re-encodes queries
and reconstructs its native prior, then averages confidence-weighted source
mask probabilities through exactly the same visible contribution operators.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('output/scannet_source_vote')
CONFIG = Path('configs/pipeline/scannet_cold_completion')
PYTHON = '/root/miniconda3/envs/cybersim_agent/bin/python'


def run_scene(name, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch
    from torch.nn import functional as F
    from radio_gs.models.native_query_encoders import load_siglip_text
    from radio_gs.pipeline.native_field import score_scene_field
    from radio_gs.querying.cold_category import categorical_prior
    from radio_gs.querying.scannet_runtime import declared_source_views
    torch.set_num_threads(4)
    scene = json.loads((CONFIG/f'{name}.scene.json').read_text())
    queries = json.loads(Path(f'configs/consolidated/scannet_native/{name}.queries.json').read_text())['queries']
    texts = [q['text'] for q in queries]
    out = ROOT/name
    out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        encoder, tokenizer = load_siglip_text(scene['text_encoder'], 'cuda')
        tokens = tokenizer(texts, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to('cuda')
        q = F.normalize(F.normalize(encoder(**tokens).pooler_output.float(), dim=-1).half().float(), dim=-1)
        del encoder
        field = torch.load(scene['native_field'], map_location='cpu', weights_only=False)
        observed = field['supported'].reshape(-1).cuda()
        prior_logits = categorical_prior(score_scene_field(scene, field, q, 'cuda').cuda(), scene['native_category_prior']['temperature'])
        logits = []
        for start in range(0, len(texts), 4):
            prior = prior_logits[:, start:start+4].sigmoid()
            numerator, mass, visibility = [torch.zeros_like(prior) for _ in range(3)]
            views = declared_source_views(observed, scene, texts[start:start+4])
            for view in views:
                weight = (view.positive | view.negative).float()*view.confidence
                numerator += torch.sparse.mm(view.adjoint, view.target*weight)
                mass += torch.sparse.mm(view.adjoint, weight)
                visibility += torch.sparse.mm(view.adjoint, view.visible.float())
            # Same0.1 prior-to-visibility ratio as the iterative solver. No
            # extra query labels, thresholds, teacher call or tuning sweep.
            value = (numerator+.1*visibility*prior)/(mass+.1*visibility).clamp_min(1e-8)
            logits.append(torch.where(mass > 0, torch.logit(value.clamp(1e-6, 1-1e-6)), prior_logits[:, start:start+4]))
            del views
        values = torch.cat(logits, dim=1)
        torch.save(dict(source_evidence_used=True, queries=queries, texts=texts, category_logits=values.cpu(), scores=values.cpu(), native_logits=prior_logits.cpu(), semantic_geometry=scene['geometry'], semantic_observed=observed.cpu(), output_contract='same_evidence_single_pass_weighted_vote'), out/'typed_readout.pt')
    subprocess.run([PYTHON, 'scripts/research/evaluate_scannet_runtime.py', '--readout', str(out/'typed_readout.pt'), '--evaluation', f'configs/consolidated/scannet_native/{name}.evaluation.json', '--output', str(out/'evaluation'), '--point-only'], env=dict(os.environ, PYTHONPATH='.'), check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene')
    p.add_argument('--gpu', default='0')
    p.add_argument('--gpus', default='0,1,2,3,4,5')
    a = p.parse_args()
    if a.scene:
        run_scene(a.scene, a.gpu)
        return
    scenes = sorted(p.name.removesuffix('.scene.json') for p in CONFIG.glob('*.scene.json'))
    gpus = a.gpus.split(',')
    ROOT.mkdir(parents=True, exist_ok=True)
    def worker(index):
        gpu = gpus[index]
        for name in scenes[index::len(gpus)]:
            out = ROOT/name
            out.mkdir(parents=True, exist_ok=True)
            with (out/'run.log').open('w') as log:
                subprocess.run([PYTHON, __file__, '--scene', name, '--gpu', str(gpu)], env=dict(os.environ, PYTHONPATH='.'), stdout=log, stderr=subprocess.STDOUT, check=True)
            print(name, 'vote evaluated', flush=True)
    with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:
        list(pool.map(worker, range(len(gpus))))
    rows = {name: {arm: json.loads((root/name/'evaluation/report.json').read_text())['point_metrics']['source'] for arm, root in [('vote', ROOT), ('solver', Path('output/scannet_cold_completion'))]} for name in scenes}
    macro = {arm: {s: sum(r[arm][s]['miou'] for r in rows.values())/len(rows) for s in ['19', '15', '10']} for arm in ['vote', 'solver']}
    (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows, macro=macro, same_geometry_prior_teacher_evidence=True, solver_steps=60, vote_passes=1, independent_test=False), indent=2)+'\n')
    print(json.dumps(macro, indent=2), flush=True)


if __name__ == '__main__':
    main()
