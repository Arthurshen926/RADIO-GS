"""Fixed full-cohort native identity transfer; separate prediction and scoring."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from radio_gs.query_field.identity_evidence import IDENTITY_CONTROLS, image_identity_controls
from radio_gs.query_field.storage import load_field, sha256

COHORT = Path('configs/query_field/five_benchmark_v1/cohort.json')
ROOT = Path('output/query_field_20261008/native_identity_transfer_v1')
PROTOCOL = Path('docs/experiments/2026-10-08-native-identity-transfer-protocol.md')
CODE = [Path(__file__), Path('radio_gs/query_field/identity_evidence.py'),
        Path('radio_gs/query_field/field.py'), Path('radio_gs/query_field/adapters.py'),
        Path('radio_gs/query_field/storage.py')]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def receipt(path):
    return dict(path=str(path), sha256=sha256(path))


def check(entry):
    if sha256(entry['path']) != entry['sha256']:
        raise ValueError('changed artifact: ' + entry['path'])


def source_guard(prefixes):
    prefixes = [Path(p).resolve() for p in prefixes]
    attempts = []

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        p = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        if any(p == root or root in p.parents for root in prefixes):
            attempts.append(str(p))
            raise PermissionError('source/evaluator access forbidden: ' + str(p))
    sys.addaudithook(guard)
    return attempts


def encode(cohort, device):
    from radio_gs.query_field.adapters import NativeQueryEncoder
    public = Path(cohort['uqis_package']) / 'method'
    queries_path = public / 'queries/image.json'
    queries = read(queries_path)['queries']
    if len(queries) != 67:
        raise ValueError('image query denominator changed')
    ROOT.mkdir(parents=True, exist_ok=False)
    write(ROOT / 'design.json', dict(protocol=receipt(PROTOCOL), code=[receipt(p) for p in CODE],
          cohort=receipt(COHORT), queries=receipt(queries_path), arms=list(IDENTITY_CONTROLS),
          training=False, default_changed=False, evaluator_used_for_prediction=False))
    attempts = source_guard([Path(cohort['uqis_package']) / 'evaluator'])
    encoder = NativeQueryEncoder(cohort['encoder'], device)
    records = []
    for query in queries:
        path = public / query['crop_rgb_path']
        if sha256(path) != query['crop_rgb_sha256']:
            raise ValueError('query image changed')
        with Image.open(path) as image:
            packet = encoder.image(image.convert('RGB'))
        target = ROOT / 'encoded' / (query['query_id'] + '.pt')
        target.parent.mkdir(exist_ok=True)
        torch.save(dict(semantic=packet.semantic.cpu(), local=packet.local.cpu()), target)
        records.append(dict(query_id=query['query_id'], scene=query['scene_id'], input=receipt(path),
                            encoded=receipt(target)))
        print('encoded', query['query_id'], tuple(packet.local.shape), flush=True)
    write(ROOT / 'encoding.json', dict(queries=records, encoder=cohort['encoder'],
                                       forbidden_open_attempts=attempts))


@torch.no_grad()
def predict(cohort, device):
    design = read(ROOT / 'design.json')
    for item in [design['protocol'], design['cohort'], design['queries']] + design['code']:
        check(item)
    encoding = read(ROOT / 'encoding.json')
    attempts = source_guard([Path(cohort['uqis_package']) / 'evaluator'] +
        [Path(j['output']) / 'observations' for j in cohort['construction']])
    baseline = Path(cohort['root']) / 'predictions/uqis'
    scenes = []
    for job in cohort['uqis']:
        scene = job['name']
        record_path = ROOT / 'receipts' / (scene + '.json')
        if record_path.exists():
            previous = read(record_path)
            if previous['field']['sha256'] != sha256(Path(job['field']) / 'field.pt'):
                raise ValueError('resume field changed')
            for row in previous['queries']:
                check(row['prediction'])
            scenes.append(receipt(record_path))
            continue
        field, manifest = load_field(job['field'], device)
        parent_receipts = {r['query_id']: r for r in read(baseline / 'receipts' / (scene + '.json'))['queries']}
        rows = []
        for query in [q for q in encoding['queries'] if q['scene'] == scene]:
            started = time.monotonic()
            check(query['encoded'])
            encoded = torch.load(query['encoded']['path'], map_location='cpu', weights_only=True)
            outputs, diagnostics = image_identity_controls(field, **encoded)
            qid = query['query_id']
            parent_path = baseline / 'gaussians' / (qid + '.pt')
            if sha256(parent_path) != parent_receipts[qid]['gaussian_sha256']:
                raise ValueError('saved parent changed')
            parent = torch.load(parent_path, map_location='cpu', weights_only=True)
            replay_error = float((outputs['parent']['membership'].cpu() - parent['membership']).abs().max())
            if replay_error != 0 or not torch.equal(outputs['parent']['candidates'].cpu(), parent['candidates']):
                raise ValueError(f'parent replay failed {qid}: {replay_error}')
            if not torch.equal(outputs['parent']['membership'].cpu() > field.threshold, parent['selected']):
                raise ValueError('parent selection replay failed')
            target = ROOT / 'predictions' / (qid + '.pt')
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            payload = {arm: {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                       for arm, data in outputs.items()}
            torch.save(payload, target)
            rows.append(dict(query_id=qid, scene=scene, prediction=receipt(target),
                encoding=query['encoded'], parent=receipt(parent_path), replay_max_abs_error=replay_error,
                diagnostics=diagnostics, seconds=time.monotonic() - started))
            print('predicted', scene, qid, {k: v['chosen'] for k, v in outputs.items()}, flush=True)
        if sha256(Path(job['field']) / 'field.pt') != manifest['files']['field.pt']['sha256']:
            raise ValueError('field changed during prediction')
        write(record_path, dict(scene=scene, field=receipt(Path(job['field']) / 'field.pt'),
              manifest=receipt(Path(job['field']) / 'manifest.json'), queries=rows,
              source_access=False, evaluator_access=False, forbidden_open_attempts=attempts))
        scenes.append(receipt(record_path))
        del field, outputs, payload, parent
        gc.collect()
        torch.cuda.empty_cache()
    write(ROOT / 'predictions_complete.json', dict(scenes=scenes, query_count=sum(
        len(read(r['path'])['queries']) for r in scenes), design=receipt(ROOT / 'design.json'),
        encoding=receipt(ROOT / 'encoding.json'), evaluator_opened=False))


def score(cohort):
    from scipy.spatial import cKDTree
    from radio_gs.benchmarks.uqis.metrics import average_precision, scene_clustered_bootstrap
    complete = read(ROOT / 'predictions_complete.json')
    if complete['query_count'] != 67 or len(complete['scenes']) != 9:
        raise ValueError('prediction cohort incomplete')
    check(complete['design']); check(complete['encoding'])
    for record in complete['scenes']:
        check(record)
        for q in read(record['path'])['queries']:
            check(q['prediction']); check(q['encoding']); check(q['parent'])
    # No evaluator labels are opened until every fixed prediction is verified.
    package = Path(cohort['uqis_package'])
    targets = read(package / 'evaluator/targets.json')
    lookup = {t['queries']['image']: t for t in targets['targets']}
    parent_evaluation = read(Path(cohort['root']) / 'predictions/uqis/evaluation.json')
    parent_ious = {r['query_id']: r['iou'] for r in parent_evaluation['explicit_selection']['per_query']}
    rows = []
    for record in complete['scenes']:
        data = read(record['path']); scene = data['scene']
        check(data['field']); check(data['manifest'])
        state = torch.load(data['field']['path'], map_location='cpu', weights_only=True)
        domain = next(d for d in targets['scene_domains'] if d['scene_id'] == scene)
        xyz = np.load(package / 'method' / domain['mesh_xyz_path'])
        labels = np.load(package / 'evaluator' / domain['mesh_instance_ids_path'])
        distance, indices = cKDTree(state['xyz'].numpy()).query(xyz, workers=4)
        anchor_distance, anchor_index = cKDTree(xyz).query(state['xyz'].numpy(), workers=4)
        threshold = read(data['manifest']['path'])['spec']['threshold']
        for query in data['queries']:
            qid = query['query_id']; target = lookup[qid]
            truth = labels == target['instance_id']
            predictions = torch.load(query['prediction']['path'], map_location='cpu', weights_only=True)
            for arm in IDENTITY_CONTROLS:
                p = predictions[arm]; values = p['membership'].numpy()[indices]
                selected = values > threshold
                tp = int((selected & truth).sum()); fp = int((selected & ~truth).sum()); fn = int((~selected & truth).sum())
                chosen = p['chosen']
                row = dict(scene=scene, query_id=qid, arm=arm, iou=tp / max(tp + fp + fn, 1),
                    ap=average_precision(values, truth), precision=tp / max(tp + fp, 1),
                    recall=tp / max(tp + fn, 1), chosen=chosen, empty=not bool(selected.any()),
                    anchor_correct=bool(chosen >= 0 and anchor_distance[chosen] <= 1e-6
                                        and labels[anchor_index[chosen]] == target['instance_id']))
                if arm == 'parent' and abs(row['iou'] - parent_ious[qid]) > 1e-12:
                    raise ValueError('parent metric replay differs')
                rows.append(row)
        del state
    metrics = ['iou', 'ap', 'precision', 'recall', 'anchor_correct']
    scenes = sorted({r['scene'] for r in rows})
    per_scene = {arm: {s: {k: float(np.mean([r[k] for r in rows if r['arm'] == arm and r['scene'] == s]))
                          for k in metrics} for s in scenes} for arm in IDENTITY_CONTROLS}
    macro = {a: {k: float(np.mean([v[k] for v in by_scene.values()])) for k in metrics}
             for a, by_scene in per_scene.items()}
    paired = {a: {k: scene_clustered_bootstrap({s: per_scene[a][s][k] - per_scene['parent'][s][k]
                                               for s in scenes}) for k in ['iou', 'ap']}
              for a in IDENTITY_CONTROLS if a != 'parent'}
    report = dict(complete=True, query_count=67, scene_count=9, arms=list(IDENTITY_CONTROLS),
        macro=macro, per_scene=per_scene, paired_difference=paired, per_query=rows,
        anchor_correct_counts={a: sum(r['anchor_correct'] for r in rows if r['arm'] == a) for a in IDENTITY_CONTROLS},
        empty_counts={a: sum(r['empty'] for r in rows if r['arm'] == a) for a in IDENTITY_CONTROLS},
        predictions_complete=receipt(ROOT / 'predictions_complete.json'),
        evaluator=receipt(package / 'evaluator/targets.json'), parent_metrics_replayed=True,
        independent_final_test=False, default_changed=False, accuracy_promotion=False,
        scope='fixed-field image identity intervention only; not a new five-task result')
    write(ROOT / 'evaluation.json', report)
    print(json.dumps(dict(macro=macro, paired=paired, anchor_counts=report['anchor_correct_counts']), indent=2))


def cold(field_path, fixture_path, deny, device):
    attempts = source_guard(deny)
    field, _ = load_field(field_path, device)
    fixture = torch.load(fixture_path, map_location='cpu', weights_only=True)
    actual, _ = image_identity_controls(field, fixture['semantic'], fixture['local'])
    for arm in IDENTITY_CONTROLS:
        expected = fixture['expected'][arm]
        for key in ['membership', 'candidates', 'identity_scores']:
            if not torch.equal(actual[arm][key].cpu(), expected[key]):
                raise ValueError('cold replay differs: ' + arm + '/' + key)
        if actual[arm]['chosen'] != expected['chosen']:
            raise ValueError('cold chosen identity differs')
    print(json.dumps(dict(arms=list(IDENTITY_CONTROLS), exact_replay=True,
                         forbidden_open_attempts=attempts, enforcement='Python file-open audit')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['encode', 'predict', 'score', 'cold'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--field'); parser.add_argument('--fixture')
    parser.add_argument('--deny', action='append', default=[])
    args = parser.parse_args(); torch.set_num_threads(4)
    if args.stage == 'cold':
        cold(args.field, args.fixture, args.deny, args.device)
    else:
        cohort = read(COHORT)
        if args.stage == 'score':
            score(cohort)
        else:
            {'encode': encode, 'predict': predict}[args.stage](cohort, args.device)
