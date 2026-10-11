"""Change only supervision sampling in the sealed canonical membership fit."""
import argparse
import copy
from pathlib import Path
import subprocess
import torch
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes
from scripts.research.run_query_field_canonical_subset_reference import ROOT as PARENT, STAGE_CODE

ROOT = Path('output/query_field_20261011/canonical_sampling_v1')
PROTOCOL = Path('docs/experiments/2026-10-11-canonical-sampling-protocol.md')
ARMS = ['seed_addition', 'dense_joint', 'sparse_joint']


def sampling_mask(view_index):
    phase = view_index % 4
    ys = ((torch.arange(16) + .5) * 192 / 16).long() * 2 + phase // 2
    xs = ((torch.arange(16) + .5) * 256 / 16).long() * 2 + phase % 2
    mask = torch.zeros(384, 512, dtype=torch.bool); mask[ys[:, None], xs[None, :]] = True
    return mask


def sampled_state(prior, state):
    result = {}
    for qi in range(prior.shape[1]):
        if qi not in state:
            result[qi] = dict(view_indices=[], confidence=[], negative=[],
                              start_membership=prior[:, qi].clone(), support_cost=torch.zeros_like(prior[:, qi]))
            continue
        s = state[qi]; r = dict(s); r['negative'] = []
        for vi, negative in zip(s['view_indices'], s['negative']):
            phase = vi % 4; mask = sampling_mask(vi)[phase//2::2, phase%2::2].flatten()
            if negative.shape != mask.shape or negative.dtype != torch.bool: raise ValueError('incorrect source phase')
            r['negative'].append(negative & mask)
        result[qi] = r
    return result


def prepare():
    assert read(PARENT / 'summary.json')['complete']
    ROOT.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(read(PARENT / 'prediction_config.json'))
    cfg.update(semantic_controls=ARMS, status='adaptive source-assisted supervision sampling attribution')
    records = []
    for j in cfg['jobs']:
        scene = j['scene']; out = ROOT / 'inputs' / scene; out.mkdir(parents=True)
        p = PARENT / 'predictions' / j['name']; inp = read(p / 'identity_evidence/inputs.json')
        response = torch.load(inp['response'], weights_only=True)
        state = torch.load(p / 'admitted_optimization_state.pt', weights_only=True)
        new = sampled_state(response['semantic_probability'], state)
        torch.save(new, out / 'optimization_state.pt')
        frames = read(inp['scene'])['source_frames']
        track = dict(seed_frame=None, frames={f: dict(present=False, positive_allowed=sampling_mask(vi),
                     negative_allowed=sampling_mask(vi), scope_reliability=1.) for vi, f in enumerate(frames)})
        torch.save(track, out / 'sampling_permission.pt')
        inp['tracks'] = [str(out / 'sampling_permission.pt')] * len(response['texts'])
        write(out / 'inputs.json', inp)
        paths = [p / 'predictions.pt', p / 'prediction.json', p / 'admitted_optimization_state.pt',
                 p / 'admitted' / scene / 'fit.json', p / 'identity_evidence/inputs.json',
                 out / 'optimization_state.pt', out / 'sampling_permission.pt', out / 'inputs.json']
        # Parent design seals all source geometry, queries, images' proposal
        # artifacts and native descriptor inputs; new identity evidence is sealed here.
        paths += list((p / 'identity_evidence').glob('identity_*.pt'))
        records += [dict(scene=scene, **receipt(path)) for path in paths]
    write(ROOT / 'prediction_config.json', cfg)
    write(ROOT / 'design.json', dict(configuration=receipt(ROOT / 'prediction_config.json'),
        protocol=receipt(PROTOCOL), parent_design=receipt(PARENT / 'design.json'),
        code=[receipt(p) for p in [__file__, *STAGE_CODE]], inputs=records,
        implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        all_parent_scores_seen=True, intervention='256 uniformly spaced supervised rays per view; all else fixed',
        primary='sparse_joint', comparator='dense_joint', default_promoted=False, whole_five_task=False))


def design(scene=None):
    d = read(ROOT / 'design.json')
    for r in d['code'] + [d['configuration'], d['protocol'], d['parent_design']]: check(r)
    for r in d['inputs']:
        if scene is None or r['scene'] == scene: check(r)
    return d


def predict(index):
    cfg = read(ROOT / 'prediction_config.json'); j = cfg['jobs'][index]; scene = j['scene']; design(scene)
    inp = ROOT / 'inputs' / scene; out = ROOT / 'predictions' / j['name']; out.mkdir(parents=True, exist_ok=False)
    execute(['-m', 'radio_gs.pipeline.subset', '--inputs', str(inp / 'inputs.json'), '--key', scene,
        '--output-base', str(out / 'sparse'), '--mode', 'admission', '--scope-initialization', 'source',
        '--pixel-loss', 'balanced', '--support-prior', 'footprint', '--raster-scale', '2', '--steps', '240',
        '--optimization-state-in', str(inp / 'optimization_state.pt'),
        '--optimization-state-out', str(out / 'optimization_state.pt')], out / 'fit.log')
    old = torch.load(PARENT / 'predictions' / j['name'] / 'predictions.pt', weights_only=True)
    result = torch.load(out / 'sparse' / scene / 'typed_readout.pt', weights_only=True)
    state = torch.load(out / 'optimization_state.pt', weights_only=True)
    expected = torch.load(inp / 'optimization_state.pt', weights_only=True)
    original_state = torch.load(PARENT / 'predictions' / j['name'] / 'admitted_optimization_state.pt', weights_only=True)
    assert state.keys() == original_state.keys()
    for qi, s in state.items():
        ref = expected[qi]; original = original_state[qi]
        assert s['view_indices'] == ref['view_indices'] == original['view_indices']
        assert s['confidence'] == ref['confidence'] == original['confidence']
        for k in ['start_membership', 'support_cost']:
            assert torch.equal(s[k], ref[k]) and torch.equal(s[k], original[k])
        for n, r in zip(s['negative'], ref['negative']): assert torch.equal(n, r)
    assert result['texts'] == old['texts'] and torch.equal(result['selection_observed'], old['observed'])
    members = dict(seed_addition=old['membership_by_control']['seed_addition'],
                   dense_joint=old['membership_by_control']['admitted_joint'], sparse_joint=result['selection_scores'])
    for qi in range(len(old['texts'])):
        if qi not in state:
            assert torch.equal(members['sparse_joint'][:, qi], old['membership_by_control']['semantic_initial'][:, qi])
    sets = {a: v > .5 for a, v in members.items()}
    old.update(scores={a: old['scores']['semantic_initial'] for a in ARMS}, membership_by_control=members,
               selected_by_control=sets, selected=sets['sparse_joint'])
    torch.save(old, out / 'predictions.pt')
    write(out / 'prediction.json', dict(configuration=cfg, job=j, predictions_sha256=receipt(out / 'predictions.pt')['sha256'],
        queries=len(old['texts']), target_labels_opened=False, source_access=True, transient_query_fitting=True,
        initialization_cost_confidence_view_order_exact=True, negative_labels_only_restricted=True,
        fit=receipt(out / 'sparse' / scene / 'fit.json'), state=receipt(out / 'optimization_state.pt'),
        default_promoted=False, persistent_field_updated=False))


def run(gpus):
    d = design(); cfg = read(d['configuration']['path'])
    def pred(i, gpu):
        execute(['-m', 'scripts.research.run_query_field_canonical_sampling', 'predict', '--index', str(i)], ROOT / 'logs' / f'predict_{i}.log', gpu)
    lanes(list(range(4)), gpus, pred)
    records = []
    for j in cfg['jobs']:
        path = ROOT / 'predictions' / j['name'] / 'prediction.json'; r = read(path)
        check(dict(path=str(path.with_name('predictions.pt')), sha256=r['predictions_sha256']))
        records.append(receipt(path))
    write(ROOT / 'predictions_complete.json', dict(records=records, queries=68, controls=3))
    def score(j, gpu):
        execute(['-m', 'radio_gs.query_field.public_lerf', 'score', '--config', str(ROOT / 'prediction_config.json'), '--job', j['name'],
                 '--output', str(ROOT / 'predictions')], ROOT / 'logs' / ('score_' + j['scene'] + '.log'), gpu)
    lanes(cfg['jobs'], gpus, score)
    per = {a: {} for a in ARMS}
    for j in cfg['jobs']:
        metrics = read(ROOT / 'predictions' / j['name'] / 'evaluation/evaluation.json')['metrics']
        previous = read(PARENT / 'predictions' / j['name'] / 'evaluation/evaluation.json')['metrics']
        for a, m in metrics.items():
            for k in ['iou', 'ap', 'localization', 'observations']: assert m[k] == previous['semantic_initial'][k]
            if a != 'sparse_joint':
                name = 'admitted_joint' if a == 'dense_joint' else a
                for k in ['miou', 'count', 'missing']: assert m['selected'][k] == previous[name]['selected'][k]
            per[a][j['scene']] = m['selected']['miou']
    result = dict(complete=True, per_scene=per, macro={a: sum(v.values()) / 4 for a, v in per.items()},
                  source_assisted=True, adaptive_development=True, predictions_before_scoring=True, default_promoted=False, whole_five_task=False)
    write(ROOT / 'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'predict', 'run']); p.add_argument('--index', type=int); p.add_argument('--gpus', default='2')
    a = p.parse_args(); torch.set_num_threads(4)
    if a.stage == 'prepare': prepare()
    elif a.stage == 'predict': predict(a.index)
    else: run(list(map(int, a.gpus.split(','))))
