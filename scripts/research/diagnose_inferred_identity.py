"""Post-evaluation attribution; evaluator labels never enter a query reader."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from scripts.research.fit_typed_referent_binding import ROOT, write
from scripts.research.summarize_typed_binding import BASE


def main():
    package = Path('output/scannet_uqis_9_v0_3_1_candidate')
    targets = json.loads((package / 'evaluator/targets.json').read_text())
    labels = {d['scene_id']: np.load(package / 'evaluator' / d['mesh_instance_ids_path'])
              for d in targets['scene_domains']}
    conditions = {name: json.loads((root / 'uqis/identity_scope/result.json').read_text())
        for name, root in [('v9', BASE), ('native_bridge', ROOT / 'native_bridge/system'),
                           ('semantic_family', ROOT / 'semantic_family/system')]}
    rows_by_condition = {name: {q['query_id']: q for q in value['explicit_selection']['per_query']}
                         for name, value in conditions.items()}
    recipe = json.loads((ROOT / 'native_bridge_recipe.json').read_text())
    rows = []
    for job in recipe['jobs']:
        if job['benchmark'] != 'uqis':continue
        scene = job['name'];membership = sparse.load_npz(Path(job['scene']).parent / 'membership.npz').tocsc()
        runs = {name: json.loads((ROOT / name / 'system/uqis' / scene / 'query/registered/runs' / (scene + '.json')).read_text())
                for name in ['native_bridge', 'semantic_family']}
        details = {name: {r['query_id']: r for r in run['query_details']} for name, run in runs.items()}
        for qid, row in rows_by_condition['native_bridge'].items():
            if row['scene_id'] != scene or row['modality'] not in ['text', 'image']:continue
            target = row['instance_id'];detail = details['native_bridge'][qid]['inferred_anchor_membership']
            seed = detail['seed_region'];vertices = membership[:, seed].indices
            seed_fraction = float(np.mean(labels[scene][vertices] == target))
            anchor = detail['anchor_vertex'];anchor_correct = bool(labels[scene][anchor] == target)
            old = rows_by_condition['v9'][qid];delta = row['iou'] - old['iou']
            record = dict(query_id=qid, scene=scene, target=target, modality=row['modality'],
                seed=seed, seed_target_fraction=seed_fraction, seed_majority_correct=seed_fraction > .5,
                anchor=anchor, anchor_label=int(labels[scene][anchor]), anchor_correct=anchor_correct,
                old={k: old[k] for k in ['iou', 'purity', 'coverage']},
                current={k: row[k] for k in ['iou', 'purity', 'coverage']}, delta=delta)
            if row['modality'] == 'text':
                other = details['semantic_family'][qid]['instance_identity']['seed_region']
                record['lexical_seed_fraction'] = float(np.mean(labels[scene][membership[:, other].indices] == target))
                record['lexical_iou'] = rows_by_condition['semantic_family'][qid]['iou']
            rows.append(record)
    summary = {}
    for modality in ['text', 'image']:
        subset = [r for r in rows if r['modality'] == modality]
        negative = [r for r in subset if r['delta'] < -.01]
        summary[modality] = dict(queries=len(subset), correct_majority_seeds=sum(r['seed_majority_correct'] for r in subset),
            correct_anchors=sum(r['anchor_correct'] for r in subset),
            correct_majority_but_wrong_anchor=sum(r['seed_majority_correct'] and not r['anchor_correct'] for r in subset),
            wrong_majority_but_correct_anchor=sum(not r['seed_majority_correct'] and r['anchor_correct'] for r in subset),
            regressions_over_1pp=len(negative),
            wrong_anchor_regressions=sum(not r['anchor_correct'] for r in negative),
            correct_anchor_lower_coverage_regressions=sum(r['anchor_correct'] and r['current']['coverage'] < r['old']['coverage'] for r in negative),
            correct_anchor_lower_purity_regressions=sum(r['anchor_correct'] and r['current']['purity'] < r['old']['purity'] for r in negative))
    text = [r for r in rows if r['modality'] == 'text']
    summary['lexical_text'] = dict(native_correct_majority=sum(r['seed_majority_correct'] for r in text),
        lexical_correct_majority=sum(r['lexical_seed_fraction'] > .5 for r in text),
        lost_correct=sum(r['seed_majority_correct'] and r['lexical_seed_fraction'] <= .5 for r in text),
        gained_correct=sum(not r['seed_majority_correct'] and r['lexical_seed_fraction'] > .5 for r in text))
    report = dict(status='evaluation-only diagnostic, no per-query routing or label-dependent repair',
        summary=summary, largest_regressions=sorted(rows, key=lambda r: r['delta'])[:10], rows=rows)
    write(ROOT / 'identity_diagnosis.json', report);print(json.dumps(summary, indent=2))


if __name__ == '__main__':main()
