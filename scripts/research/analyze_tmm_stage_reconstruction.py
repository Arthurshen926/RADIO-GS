"""Close a sealed stage study using small reports only; never select a solver.

Source objective reductions and benchmark changes have different units. The
class decomposition exactly sums to scene-macro ScanNet mIoU, rather than
silently substituting a pooled point metric or a class-macro mean.
"""
import argparse
import json
import statistics
from pathlib import Path

from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from scripts.research.replay_tmm_response import STUDY, POLICY, sha


def read(path):
    return json.loads(Path(path).read_text())


def class_contributions(rows, arm):
    """Per-class additive contributions to the original scene-macro delta."""
    totals = {}
    for row in rows:
        original = row['per_class']['diagonal']['19']['per_class']
        changed = row['per_class'][arm]['19']['per_class']
        valid = [key for key, value in original.items() if value['iou'] is not None]
        assert valid == [key for key, value in changed.items() if value['iou'] is not None]
        for key in valid:
            before, after = original[key], changed[key]
            assert before['gt_count'] == after['gt_count']
            item = totals.setdefault(key, dict(class_id=int(key), name=before['name'],
                present_scenes=0, contribution_to_macro_delta_pp=0., per_scene=[]))
            delta = 100 * (after['iou'] - before['iou'])
            item['present_scenes'] += 1
            item['contribution_to_macro_delta_pp'] += delta / (len(valid) * len(rows))
            item['per_scene'].append(dict(scene=row['job']['name'], delta_iou_pp=delta,
                valid_class_count=len(valid)))
    expected = 100 * statistics.mean(row['metrics'][arm]['miou_19'] -
        row['metrics']['diagonal']['miou_19'] for row in rows)
    assert abs(sum(x['contribution_to_macro_delta_pp'] for x in totals.values()) - expected) < 1e-10
    return sorted(totals.values(), key=lambda x: x['contribution_to_macro_delta_pp'])


def figure(records, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(lerf='#0072B2', nvos='#009E73', scannet='#D55E00')
    names = dict(lerf='LERF: semantic IoU', nvos='NVOS: IoU', scannet='ScanNet: mIoU19')
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True)
    for ax, family in zip(axes, colors):
        rows = [r for r in records if r['family'] == family]
        ax.axhline(0, color='.55', linewidth=.8, linestyle='--')
        ax.scatter([r['squared_source_reduction_percent'] for r in rows],
            [r['squared_task_delta_pp'] for r in rows], color=colors[family], s=36)
        ax.set_title(names[family] + f" (n={len(rows)})", fontsize=10)
        ax.set_xlabel('Source objective reduction (%)', fontsize=9)
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Task score change (pp)', fontsize=9)
    fig.suptitle('Rendered squared reconstruction versus diagonal consensus', fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='stage_reconstruction_v4')
    parser.add_argument('--output', default='stage_reconstruction_v4_analysis')
    args = parser.parse_args()
    root = STUDY / args.run
    summary, manifest, seal, evaluation = [read(root / name) for name in
        ('summary.json', 'manifest.json', 'predictions_complete.json', 'evaluation_manifest.json')]
    assert summary['complete'] and summary['all_prediction_jobs'] == seal['jobs'] == 31
    assert seal['no_target_gt_read'] and evaluation['all_inputs_and_predictions_checked']
    assert evaluation['prediction_seal_sha256'] == sha(root / 'predictions_complete.json')
    assert len(seal['reports']) == 31
    reports = []
    for path, digest in seal['reports'].items():
        assert sha(path) == digest, path
        reports.append(read(path))
    assert all(not r['ground_truth_read'] for r in reports)
    sources = {**manifest['source_hashes'], **evaluation['scoring_source_hashes']}
    assert all(sha(path) == digest for path, digest in sources.items())
    feedback_path = Path('paper/tmm_response') / f'{args.run}_feedback.json'
    feedback = read(feedback_path)
    assert feedback['summary_sha256'] == sha(root / 'summary.json')
    released = read(root / 'ram_released.json')
    assert released['removed_own_tmpfs_bank'] and not Path(manifest['prediction_ram_directory']).exists()
    result_by_job = {(r['job']['benchmark'], r['job']['name']): r for r in summary['rows']}
    metrics = dict(lerf='semantic_iou', nvos='iou', scannet='miou_19')
    records, compatibility = [], {}
    for report in reports:
        family, name = report['job']['benchmark'], report['job']['name']
        if family == 'uqis':
            assert not report['intervention_applied'] and report['compatibility_only']
            assert len({v['sha256'] for v in report['prediction_arrays'].values()}) == 1
            continue
        check = report['compatibility']
        assert check['passed'] and check['half_threshold_crossings'] == 0
        compatibility.setdefault(family, []).append(check)
        result = result_by_job[(family, name)]
        values, arms = result['metrics'], report['arms']
        squared_reduction = 100 * (1 - arms['rendered_squared']['source_squared_objective'] /
            arms['diagonal']['source_squared_objective'])
        record = dict(family=family, name=name, metric=metrics[family],
            squared_source_reduction_percent=squared_reduction,
            squared_task_delta_pp=100*(values['rendered_squared'][metrics[family]] -
                values['diagonal'][metrics[family]]),
            huber_task_delta_pp=100*(values['rendered_huber'][metrics[family]] -
                values['diagonal'][metrics[family]]),
            huber_vs_squared_task_delta_pp=100*(values['rendered_huber'][metrics[family]] -
                values['rendered_squared'][metrics[family]]),
            source_objectives=arms)
        records.append(record)
    family_stats = {}
    for family in metrics:
        rows = [r for r in records if r['family'] == family]
        reductions = [r['squared_source_reduction_percent'] for r in rows]
        checks = compatibility[family]
        family_stats[family] = dict(jobs=len(rows),
            squared_objective_reduced_jobs=sum(x > 0 for x in reductions),
            source_reduction_median_percent=statistics.median(reductions),
            source_reduction_range_percent=[min(reductions), max(reductions)],
            squared_task_improved_jobs=sum(r['squared_task_delta_pp'] > 1e-8 for r in rows),
            squared_task_degraded_jobs=sum(r['squared_task_delta_pp'] < -1e-8 for r in rows),
            original_solver_max_error=max(x['maximum_response_error'] for x in checks),
            original_solver_max_mean_error=max(x['mean_response_error'] for x in checks),
            original_solver_half_crossings=sum(x['half_threshold_crossings'] for x in checks))
    scan = [r for r in summary['rows'] if r['job']['benchmark'] == 'scannet']
    classes = {arm: class_contributions(scan, arm) for arm in ('rendered_squared', 'rendered_huber')}
    persistent = tree_bytes(STUDY)
    result = dict(schema='tmm.stage_reconstruction.analysis.v1', run=args.run,
        complete=True, all_prediction_reports_verified=31,
        source_and_scorer_files_currently_match_recorded_hashes=len(sources),
        input_and_array_hash_verification='driver verified all before scoring; evaluation manifest sealed',
        arrays_rechecked_after_release=False, dense_arrays_removed=True,
        source_fit_and_task_rows=records, families=family_stats,
        scannet19_class_contributions=classes,
        class_contribution_unit='additive percentage points of scene-macro mIoU, not per-class IoU',
        paired_metric_feedback=str(feedback_path), feedback_sha256=sha(feedback_path),
        summary_sha256=sha(root / 'summary.json'),
        predictions_seal_sha256=sha(root / 'predictions_complete.json'),
        actual_set_improvement_claim=False, uqis_compatibility_only=True,
        independent_test=False, new_default_selected=False,
        conclusion='Lower source reconstruction loss is insufficient for downstream gains; no universal solver replacement selected.',
        attribution_limit='Does not isolate identity errors, incomplete masks, geometry errors or cross-view inconsistency.',
        storage=dict(persistent_study_bytes_before_analysis=persistent,
            persistent_increment_since_turn_start_bytes=persistent-2036939577,
            transient_predictions_released_bytes=summary['transient_prediction_bytes'],
            persistent_study_budget_bytes=POLICY['study_bytes']))
    out = BoundedExperiment(STUDY, args.output, POLICY)
    out.json('analysis.json', result)
    out.json('source_snapshot.json', dict(path=__file__, sha256=sha(__file__), text=Path(__file__).read_text()))
    (Path('paper/tmm_response') / f'{args.run}_analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    figure(records, Path('paper/tmm_response/figures') / 'stage_source_fit_vs_task')
    print(json.dumps(dict(families=family_stats, class_losses={arm:[(r['name'],
        r['contribution_to_macro_delta_pp']) for r in rows[:5]] for arm,rows in classes.items()},
        storage=result['storage']), indent=2))


if __name__ == '__main__':
    main()
