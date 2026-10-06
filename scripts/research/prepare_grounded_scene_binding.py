"""Attach published training-only entity labels to unchanged query encodings."""
import ast
import csv
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

import numpy as np
from scipy import sparse
from transformers import AutoTokenizer

ROOT = Path('output/grounded_scene_binding_20261006')
DATA = Path('output/real_referent_grounding_20261005')
NAMED = Path('output/query_specific_grounding_20261005')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def aligned_subwords(expression, official, indices, offsets):
    words = list(re.finditer(r"\w+|[^\w\s]", expression.lower()))
    values = [m.group() for m in words]
    aligned = {}
    for block in SequenceMatcher(None, values, official, autojunk=False).get_matching_blocks():
        aligned.update({block.b + i: words[block.a + i].span() for i in range(block.size)})
    if not all(i in aligned for i in indices):
        return [], 'unaligned_source_words'
    spans = [aligned[i] for i in indices]
    tokens = [i for i, (lo, hi) in enumerate(offsets)
              if hi > lo and any(lo < end and hi > start for start, end in spans)]
    # A partially truncated phrase does not supervise the surviving fragment.
    if not all(any(lo < end and hi > start for lo, hi in offsets if hi > lo) for start, end in spans):
        return [], 'outside_query_token_window'
    return tokens, None


def main():
    design = json.loads((DATA / 'design.json').read_text())
    ROOT.mkdir(exist_ok=True)
    for directory, source in [('maps', DATA / 'maps'), ('language', NAMED / 'language'), ('labels', DATA / 'labels')]:
        if not (ROOT / directory).exists():
            (ROOT / directory).symlink_to(source.resolve(), target_is_directory=True)
    write(ROOT / 'design.json', design)
    source = list(csv.DictReader((ROOT / 'source/ScanEnts3D_Nr3D.csv').open()))
    lookup = {(r['scan_id'], int(r['target_id']), r['utterance']): r for r in source}
    encoder = json.loads((DATA / 'maps' / design['fit'][0] / 'map.json').read_text())['native_encoder']
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    out = ROOT / 'entities';out.mkdir(exist_ok=True)
    records = []
    for split in ['fit', 'calibration', 'validation']:
        for scene in design[split]:
            rows = json.loads((ROOT / 'language' / (scene + '.json')).read_text())
            tokens = tokenizer([r['expression'] for r in rows], padding='max_length', max_length=64,
                               truncation=True, return_offsets_mapping=True)
            labels = np.load(DATA / 'labels' / (scene + '.npy'))
            classes = np.unique(labels[labels >= 0]);position = {int(v): i for i, v in enumerate(classes)}
            truth = sparse.csr_matrix((np.ones(np.sum(labels >= 0), np.float32),
                (np.flatnonzero(labels >= 0), np.searchsorted(classes, labels[labels >= 0]))), shape=(len(labels), len(classes)))
            folder = DATA / 'maps' / scene
            scopes = sparse.load_npz(folder / 'scope_selections.npz')
            membership = sparse.load_npz(folder / 'membership.npz')
            intersection = (scopes.T @ truth).toarray()
            target_size = np.asarray(truth.sum(0)).ravel()
            area = np.asarray(scopes.sum(0)).ravel()
            reward = intersection / np.maximum(area[:, None] + target_size - intersection, 1)
            purity = (membership.T @ truth).toarray() / np.maximum(np.asarray(membership.sum(0)).ravel()[:, None], 1)
            np.savez_compressed(out / (scene + '.npz'), instances=classes, reward=reward, purity=purity)
            annotations = [];counts = dict(queries=len(rows), matched_queries=0, mentions=0, target_mentions=0,
                landmark_mentions=0, annotated_mentions=0, unknown_instance=0,
                unaligned_source_words=0, outside_query_token_window=0, unavailable_scope=0)
            for index, row in enumerate(rows):
                found = lookup.get((scene, row['raw_object_id'], row['expression']))
                entities = []
                if found is not None:
                    counts['matched_queries'] += 1
                    official = ast.literal_eval(found['tokens'])
                    for indices, objects in ast.literal_eval(found['entities']):
                        counts['annotated_mentions'] += 1
                        subwords, reason = aligned_subwords(row['expression'], official, indices, tokens['offset_mapping'][index])
                        if reason:
                            counts[reason] += 1;continue
                        objects = [int(v.split('_')[0]) + 1 for v in objects]
                        if not objects or any(v not in position for v in objects):
                            counts['unknown_instance'] += 1;continue
                        columns = [position[v] for v in objects]
                        if np.any(reward[:, columns].max(0) == 0):
                            counts['unavailable_scope'] += 1;continue
                        is_target = row['raw_object_id'] + 1 in objects
                        entities.append(dict(tokens=subwords, instances=objects, columns=columns, target=is_target,
                            phrase=' '.join(official[i] for i in indices)))
                        counts['mentions'] += 1
                        counts['target_mentions' if is_target else 'landmark_mentions'] += 1
                annotations.append(entities)
            write(out / (scene + '.json'), annotations)
            records.append(dict(scene=scene, split=split, **counts))
            print(scene, counts, flush=True)
    write(ROOT / 'preparation.json', dict(source='https://scanents3d.github.io/dataset.html',
        annotation_rows=len(source), exact_query_key='scene, original object ID, unchanged original utterance',
        source_labels_are_additional_supervision=True, benchmark_annotations_read=False,
        scenes=records, groups={k: {name: sum(r[name] for r in records if r['split'] == k)
            for name in records[0] if name not in ['scene', 'split']} for k in design if k in ['fit', 'calibration', 'validation']}))


if __name__ == '__main__':
    main()
