"""Evaluate the predeclared common program on the cross-view Gaussian fields."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch

from radio_gs.query_field.contracts import QueryPacket
from radio_gs.query_field.query_program import PROGRAM, query
from radio_gs.query_field.storage import load_field, sha256
from scripts.research.run_query_field_relations import ROOT, read, write, receipt, check

OUT = ROOT / 'common_program'
PROTOCOL = Path('docs/experiments/2026-10-08-common-query-program-protocol.md')
ENCODINGS = Path('output/query_field_20261008/native_identity_transfer_v1/encoding.json')


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / 'design.json', dict(program=PROGRAM, field_arm='cross_view_anchor', protocol=receipt(PROTOCOL),
        encodings=receipt(ENCODINGS), code=[receipt(Path(__file__)),
            *[receipt(Path('radio_gs/query_field') / (m + '.py')) for m in
              ['query_program', 'identity_evidence', 'category_evidence', 'field', 'adapters', 'storage']]],
        original_identity=read(ENCODINGS)['encoder'], selection='fixed before new model scores',
        readouts=dict(lerf='same cross-view field text-union outputs', nvos='same cross-view signed outputs',
                      scannet='predeclared bounded native/member category evidence', uqis=PROGRAM)))


@torch.no_grad()
def predict():
    read(ROOT / 'predictions_complete.json')
    design = read(OUT / 'design.json')
    for r in design['code'] + [design['protocol'], design['encodings']]: check(r)
    cohort = read(ROOT / 'cross_view_anchor/cohort.json')
    public = Path(cohort['uqis_package']) / 'method'
    queries = {q['query_id']: q for q in read(public / 'queries/image.json')['queries']}
    encodings = {q['query_id']: q for q in read(ENCODINGS)['queries']}
    source = ROOT / 'cross_view_anchor/predictions/uqis'
    for folder in ['predictions', 'selections', 'gaussians', 'receipts']:
        (OUT / 'uqis' / folder).mkdir(parents=True, exist_ok=True)
    records = []
    for job in cohort['uqis']:
        destination = OUT / 'uqis/receipts' / (job['name'] + '.json')
        if destination.exists():
            records.append(receipt(destination)); continue
        field, manifest = load_field(job['field'], 'cuda')
        domain = job['public_domain']; xyz_path = public / domain['mesh_xyz_path']
        if sha256(xyz_path) != domain['mesh_xyz_sha256']: raise ValueError('public domain changed')
        _, index = cKDTree(field.xyz.cpu().numpy()).query(np.load(xyz_path), workers=4)
        original = read(source / 'receipts' / (job['name'] + '.json'))
        if original['field_sha256'] != manifest['files']['field.pt']['sha256']:
            raise ValueError('reused nonimage outputs use another field')
        rows = []
        for old in original['queries']:
            qid = old['query_id']; modality = old['modality']
            paths = {folder: OUT / 'uqis' / folder / (qid + suffix)
                     for folder, suffix in [('predictions', '.npy'), ('selections', '.npy'), ('gaussians', '.pt')]}
            if any(p.exists() for p in paths.values()): raise FileExistsError(qid)
            if modality != 'image':
                row = dict(old, query_program=PROGRAM, reused_identical_program_path=True)
                for folder, key in [('predictions', 'prediction_sha256'), ('selections', 'selection_sha256'),
                                    ('gaussians', 'gaussian_sha256')]:
                    source_path = source / folder / paths[folder].name
                    if sha256(source_path) != old[key]: raise ValueError('unchanged output changed')
                    shutil.copy2(source_path, paths[folder])
            else:
                q = queries[qid]; encoding = encodings[qid]; check(encoding['encoded'])
                if q['crop_rgb_sha256'] != encoding['input']['sha256']:
                    raise ValueError('current query differs from encoded image')
                e = torch.load(encoding['encoded']['path'], map_location='cpu', weights_only=True)
                packet = QueryPacket('image', semantic=e['semantic'].to(field.xyz.device), local=e['local'].to(field.xyz.device))
                result = query(field, packet)
                scores = result.membership_scores.cpu().numpy()
                np.save(paths['predictions'], scores[index].astype(np.float32))
                np.save(paths['selections'], result.selected.cpu().numpy()[index])
                torch.save(dict(membership=result.membership_scores.cpu(), selected=result.selected.cpu(),
                    candidates=result.candidate_indices.cpu(), candidate_weights=result.candidate_weights.cpu(),
                    native_scores=result.semantic_scores.cpu()), paths['gaussians'])
                row = dict(query_id=qid, modality='image', status='ok', query_program=PROGRAM,
                    reused_identical_program_path=False, diagnostics=result.diagnostics, encoding=encoding['encoded'])
                for folder, key in [('predictions', 'prediction_sha256'), ('selections', 'selection_sha256'),
                                    ('gaussians', 'gaussian_sha256')]: row[key] = sha256(paths[folder])
                print('common program image', job['name'], qid, flush=True)
            rows.append(row)
        write(destination, dict(scene=job['name'], field_sha256=manifest['files']['field.pt']['sha256'],
                                parent_receipt=receipt(source / 'receipts' / (job['name'] + '.json')), queries=rows))
        records.append(receipt(destination)); del field; gc.collect(); torch.cuda.empty_cache()
    verify(records)
    write(OUT / 'predictions_complete.json', dict(records=records, design=receipt(OUT / 'design.json'),
        original_cohort=receipt(ROOT / 'predictions_complete.json'), new_image_predictions=67,
        exact_nonimage_path_reused=201, category_receipt=receipt(ROOT / 'category_evidence/predictions_complete.json')))


def verify(records):
    query_ids = set()
    for entry in records:
        check(entry)
        for row in read(entry['path'])['queries']:
            query_ids.add(row['query_id'])
            for folder, suffix, key in [('predictions', '.npy', 'prediction_sha256'),
                ('selections', '.npy', 'selection_sha256'), ('gaussians', '.pt', 'gaussian_sha256')]:
                if sha256(OUT / 'uqis' / folder / (row['query_id'] + suffix)) != row[key]:
                    raise ValueError('common program output changed')
    if len(query_ids) != 268: raise ValueError('common program query cohort incomplete')


def score():
    import subprocess
    complete = read(OUT / 'predictions_complete.json'); verify(complete['records'])
    cohort = read(ROOT / 'cross_view_anchor/cohort.json')
    subprocess.run([sys.executable, '-m', 'radio_gs.benchmarks.uqis.evaluate', '--package', cohort['uqis_package'],
        '--predictions', str(OUT / 'uqis/predictions'), '--selections', str(OUT / 'uqis/selections'),
        '--output', str(OUT / 'uqis/evaluation.json')], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['prepare', 'predict', 'score'])
    args = p.parse_args(); torch.set_num_threads(4)
    {'prepare': prepare, 'predict': predict, 'score': score}[args.stage]()
