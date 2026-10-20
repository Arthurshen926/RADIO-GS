"""Report actual decision units, historical capacity and source-evidence limits."""
import csv
import json
from pathlib import Path

import numpy as np

from radio_gs.querying.response_study_adapters import read
from radio_gs.response_study.decision_evidence import whole_response_metrics
from scripts.research.audit_tmm_source_decisions import candidate_tables
from scripts.research.replay_tmm_response import STUDY, sha


def joint_metric_oracle(job, names, tables):
    """Diagnostic dominance restriction. Ground truth MUST NOT drive inference."""
    if job['benchmark'] == 'lerf':
        queries = sorted({r['query'] for r in tables[0]})
        choices = {}
        for q in queries:
            m = [{k: float(np.mean([r[k] for r in t if r['query'] == q])) for k in ('iou', 'ap')} for t in tables]
            feasible = [i for i, v in enumerate(m) if all(v[k] >= m[0][k]-1e-12 for k in m[0])]
            choices[q] = max(feasible, key=lambda i: (m[i]['iou'], -i))
        value = whole_response_metrics(tables, choices, tables[0])
        return dict(metrics=dict(semantic_iou=value['iou'], ap=value['ap']), choices=choices)
    key = 'iou' if job['benchmark'] == 'nvos' else 'miou_19'
    feasible = [i for i, v in enumerate(tables) if all(v[k] >= tables[0][k]-1e-12 for k in tables[0])]
    selected = max(feasible, key=lambda i: (tables[i][key], -i))
    return dict(metrics=tables[selected], choice=selected)


def main():
    root = STUDY/'source_decision_audit_v1'; summary = read(root/'summary.json')
    config = read(root/'manifest.json')['config']
    rows, frontier = [], []
    for row in summary['rows']:
        for p, digest in row['cached_score_hashes'].items():
            if sha(p) != digest:
                raise ValueError('historical candidate metrics changed')
        names, tables, _ = candidate_tables(row['job'], config)
        frontier.append(dict(job=row['job'], **joint_metric_oracle(row['job'], names, tables)))
    pareto = {}
    for family, arms in summary['macro'].items():
        local = [r for r in frontier if r['job']['benchmark'] == family]
        pareto[family] = {k: float(np.mean([r['metrics'][k] for r in local])) for k in local[0]['metrics']}
        for metric in arms['baseline']:
            rows.append(dict(family=family, metric=metric,
                **{a: v[metric]*100 for a, v in arms.items()},
                joint_metric_oracle=pareto[family][metric]*100))
    target_path = STUDY/'cross_task_targets_v1/summary.json'; targets = read(target_path)
    feedback = dict(complete=True, diagnostic_only=True, source_audit=summary,
        historical_joint_metric_oracle=dict(macro=pareto, rows=frontier,
            rule='label oracle restricted to candidates that do not harm ANY reported metric within a decision unit',
            uses_evaluation_labels=True, inference_policy=False, no_claim_of_uniform_achievability=True),
        independent_target_preparation=targets,
        source_hashes={str(root/'summary.json'): sha(root/'summary.json'), str(target_path): sha(target_path)},
        default_changed=False, source_candidate_replay='E02 dense candidates verified exactly; larger historical pool uses retained scoring receipts',
        uqis_fresh_inference=False)
    paper = Path('paper/tmm_response'); (paper/'artifacts').mkdir(exist_ok=True)
    (paper/'cross_task_feedback.json').write_text(json.dumps(feedback, indent=2)+'\n')
    with (paper/'artifacts/cross_task_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    table = ['\\begin{tabular}{lrrrr}', '\\toprule',
        'Metric (\\%) & v10 & Source agreement & Pool oracle & Joint oracle \\\\', '\\midrule']
    labels = {('lerf','semantic_iou'): 'LERF semantic IoU', ('nvos','iou'): 'NVOS IoU',
              ('scannet','miou_19'): 'ScanNet mIoU19', ('scannet','miou_15'): 'ScanNet mIoU15',
              ('scannet','miou_10'): 'ScanNet mIoU10'}
    for row in rows:
        if (row['family'], row['metric']) in labels:
            table.append(labels[row['family'], row['metric']]+' & '+' & '.join(
                f"{row[k]:.3f}" for k in ('baseline','full','historical_pool_oracle','joint_metric_oracle'))+' \\\\')
    table += ['\\bottomrule', '\\end{tabular}']
    (paper/'tables/cross_task_capacity.tex').write_text('\n'.join(table)+'\n')
    findings = [r for r in rows if (r['family'],r['metric']) in labels]
    print(json.dumps(dict(rows=findings, joint_oracle=pareto, target_coverage=targets['class_room_coverage']), indent=2))


if __name__ == '__main__':
    main()
