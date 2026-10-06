"""Fixed identity, two scope readers; inferred anchors are not user prompts."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from radio_gs.querying.direct_anchor_membership import DirectAnchorMembership
from scripts.research.fit_typed_referent_binding import ROOT, SOURCE, write


def evaluate_scene(scene):
    folder = SOURCE / 'maps' / scene
    xyz = np.load(folder / 'mesh_xyz.npy');labels = np.load(SOURCE / 'labels' / (scene + '.npy'))
    membership = sparse.load_npz(folder / 'membership.npz').tocsc()
    visibility = sparse.load_npz(folder / 'visibility.npz')
    frames = np.repeat(np.arange(visibility.shape[1]), np.diff(np.load(folder / 'region_offsets.npy')))
    reader = DirectAnchorMembership(membership, visibility, frames, xyz, np.load(folder / 'descriptors.npy'),
        sparse.load_npz(folder / 'transfer.npz'), 'output/real_direct_membership_20261006/real/model.joblib')
    queries = json.loads((SOURCE / 'language' / (scene + '.json')).read_text())
    hypotheses = sparse.load_npz(folder / 'scope_selections.npz').tocsc()
    stages = {}
    for stage in ['semantic', 'frozen', 'joint']:
        saved = json.loads((ROOT / stage / 'fit.json').read_text())
        stages[stage] = next(row for row in saved['validation']['scenes'] if row['scene'] == scene)['conditions']['original']
        if stage == 'semantic':
            stages['native'] = next(row for row in saved['validation']['scenes'] if row['scene'] == scene)['conditions']['native']
    selections = {};anchors = {};rows = []
    for seed in sorted({seed for values in stages.values() for seed in values['seeds']}):
        vertices = membership[:, seed].indices
        median = np.median(xyz[vertices], axis=0)
        anchor = int(vertices[np.argmin(((xyz[vertices] - median) ** 2).sum(1))])
        anchors[seed] = anchor
        if anchor not in selections:
            _, selected, _ = reader.infer(anchor);selections[anchor] = np.flatnonzero(selected)
    for stage, values in stages.items():
        for i, (query, seed) in enumerate(zip(queries, values['seeds'])):
            target = query['raw_object_id'] + 1;truth_size = int(np.count_nonzero(labels == target))
            anchor = anchors[seed];conditions = {}
            for name, selected in [('family', hypotheses[:, seed].indices), ('direct_membership', selections[anchor])]:
                intersection = int(np.count_nonzero(labels[selected] == target))
                conditions[name] = dict(iou=intersection / (len(selected) + truth_size - intersection),
                    precision=intersection / max(len(selected), 1), recall=intersection / truth_size,
                    selected_elements=len(selected))
            rows.append(dict(stage=stage, query=i, seed=seed, inferred_anchor=anchor,
                anchor_correct=bool(labels[anchor] == target), conditions=conditions,
                cached_family_iou=values['per_query_iou'][i]))
    summary = {}
    for stage in stages:
        subset = [row for row in rows if row['stage'] == stage]
        summary[stage] = dict(anchor_accuracy=float(np.mean([r['anchor_correct'] for r in subset])),
            **{condition: {metric: float(np.mean([r['conditions'][condition][metric] for r in subset]))
                for metric in ['iou', 'precision', 'recall']} for condition in ['family', 'direct_membership']})
    result = dict(scene=scene, queries=len(queries), unique_anchors=len(selections), rows=rows, summary=summary,
        original_scope_reproduction_max_error=max(abs(r['cached_family_iou'] - r['conditions']['family']['iou']) for r in rows))
    write(ROOT / 'membership_bridge' / (scene + '.json'), result)
    print(scene, summary, flush=True);return result


def main():
    design = json.loads((ROOT / 'design.json').read_text())
    declaration = dict(identity_rules=['native', 'semantic', 'frozen', 'joint'],
        scope_rules=['same existing one-hop family', 'existing v9 actual-anchor membership model'],
        inferred_anchor='nearest actual member to coordinate-wise median of the chosen source region',
        anchor_selection_uses_labels=False, images_at_query=False, threshold_changed=False,
        extra_fit=False, oracle_prompt=False,
        gate='semantic bridge exceeds previous11.110% and same semantic family each by2pp; native bridge also improves native family',
        inference_uncertainty='Automatically inferred anchor may be wrong; every query and all wrong-anchor cases retained')
    out = ROOT / 'membership_bridge';out.mkdir(exist_ok=True);write(out / 'design.json', declaration)
    with ThreadPoolExecutor(4) as pool:scenes = list(pool.map(evaluate_scene, design['validation']))
    summary = {stage: {condition: {metric: float(np.mean([r['summary'][stage][condition][metric] for r in scenes]))
        for metric in ['iou', 'precision', 'recall']} for condition in ['family', 'direct_membership']}
        for stage in scenes[0]['summary']}
    for stage in summary:
        summary[stage]['anchor_accuracy'] = float(np.mean([r['summary'][stage]['anchor_accuracy'] for r in scenes]))
        delta = np.array([r['summary'][stage]['direct_membership']['iou'] - r['summary'][stage]['family']['iou'] for r in scenes])
        rng = np.random.default_rng(20261006)
        summary[stage]['paired_delta'] = dict(mean=float(delta.mean()), per_scene=delta.tolist(),
            bootstrap95=np.quantile(delta[rng.integers(0, len(delta), (20000, len(delta)))].mean(1), [.025, .975]).tolist())
    value = summary['semantic']['direct_membership']['iou'];prior = .11109819856937975
    gate = dict(passed=value >= max(prior, summary['semantic']['family']['iou']) + .02 and
        summary['native']['direct_membership']['iou'] > summary['native']['family']['iou'],
        semantic_bridge=value, prior=prior, native_bridge=summary['native']['direct_membership']['iou'])
    write(out / 'summary.json', dict(declaration=declaration, scenes=scenes, scene_macro=summary,
        development_gate=gate, benchmark_jobs=0, independent_final_test=False))
    print('bridge gate', gate, flush=True)


if __name__ == '__main__':main()
