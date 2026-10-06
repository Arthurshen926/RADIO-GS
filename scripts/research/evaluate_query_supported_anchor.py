"""One fixed repair: carry native query evidence from a region to its anchor."""
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
from scipy import sparse

from radio_gs.querying.direct_anchor_membership import DirectAnchorMembership
from scripts.research.fit_typed_referent_binding import ROOT, SOURCE, write


def evaluate_scene(scene):
    folder = SOURCE / 'maps' / scene
    xyz = np.load(folder / 'mesh_xyz.npy')
    labels = np.load(SOURCE / 'labels' / (scene + '.npy'))
    membership = sparse.load_npz(folder / 'membership.npz').tocsc()
    visibility = sparse.load_npz(folder / 'visibility.npz')
    frames = np.repeat(np.arange(visibility.shape[1]), np.diff(np.load(folder / 'region_offsets.npy')))
    descriptors = np.load(folder / 'descriptors.npy').astype(np.float32)
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8)
    language = np.load(SOURCE / 'language' / (scene + '.npz'))['pooled'].astype(np.float32)
    language /= np.maximum(np.linalg.norm(language, axis=1, keepdims=True), 1e-8)
    response = descriptors @ language.T
    reader = DirectAnchorMembership(membership, visibility, frames, xyz, descriptors,
        sparse.load_npz(folder / 'transfer.npz'), 'output/real_direct_membership_20261006/real/model.joblib')
    queries = json.loads((SOURCE / 'language' / (scene + '.json')).read_text())
    baseline = json.loads((ROOT / 'membership_bridge' / (scene + '.json')).read_text())
    rows = []
    for old in baseline['rows']:
        if old['stage'] != 'native':continue
        index = old['query'];target = queries[index]['raw_object_id'] + 1
        _, selected, detail = reader.infer_region(old['seed'], response[:, index])
        intersection = np.count_nonzero(selected & (labels == target));size = np.count_nonzero(labels == target)
        rows.append(dict(query=index, seed=old['seed'], anchor=detail['anchor_vertex'],
            anchor_correct=bool(labels[detail['anchor_vertex']] == target),
            old_anchor_correct=old['anchor_correct'], old=old['conditions']['direct_membership'],
            current=dict(iou=float(intersection / (selected.sum() + size - intersection)),
                precision=float(intersection / max(selected.sum(), 1)), recall=float(intersection / size))))
    result = dict(scene=scene, rows=rows, old_iou=float(np.mean([r['old']['iou'] for r in rows])),
        current_iou=float(np.mean([r['current']['iou'] for r in rows])),
        old_anchor_accuracy=float(np.mean([r['old_anchor_correct'] for r in rows])),
        current_anchor_accuracy=float(np.mean([r['anchor_correct'] for r in rows])))
    write(ROOT / 'query_supported_anchor' / (scene + '.json'), result)
    print(scene, {k: v for k, v in result.items() if k != 'rows'}, flush=True)
    return result


def main():
    design = json.loads((ROOT / 'design.json').read_text())
    out = ROOT / 'query_supported_anchor';out.mkdir(exist_ok=True)
    declaration = dict(candidates=1, fit=False, parameter_scan=False,
        mechanism='same chosen source region; maximum view-balanced native query identity selects actual element; median tie break',
        prompted_anchors_changed=False, model_and_threshold_changed=False,
        motivation='opened UQIS development result exposes geometric median anchor on wrong instance',
        selection_data='four previously opened NR3D development rooms, disjoint from UQIS',
        gate='native text IoU exceeds median anchor by at least1pp and at least3 of4 scenes do not regress',
        independent_final_test=False)
    write(out / 'design.json', declaration)
    with ThreadPoolExecutor(4) as pool:scenes = list(pool.map(evaluate_scene, design['validation']))
    delta = np.array([r['current_iou'] - r['old_iou'] for r in scenes])
    summary = dict(declaration=declaration, scenes=scenes, old_iou=float(np.mean([r['old_iou'] for r in scenes])),
        current_iou=float(np.mean([r['current_iou'] for r in scenes])), delta=float(delta.mean()),
        old_anchor_accuracy=float(np.mean([r['old_anchor_accuracy'] for r in scenes])),
        current_anchor_accuracy=float(np.mean([r['current_anchor_accuracy'] for r in scenes])),
        gate_passed=bool(delta.mean() >= .01 and (delta >= 0).sum() >= 3))
    write(out / 'summary.json', summary);print('summary', {k: v for k, v in summary.items() if k != 'scenes'}, flush=True)


if __name__ == '__main__':main()
