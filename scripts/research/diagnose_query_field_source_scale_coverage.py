"""Construction-only check of default-scale extrapolation, without GT or fitting."""
from pathlib import Path

import numpy as np

from scripts.research.run_query_field_joint_elements import ROOT, frozen_design, read, write, receipt, check


def main():
    design = frozen_design(); rows = []
    for job in design['fields']:
        check(job['parent_manifest']); check(job['matched_element_training'])
        manifest = read(job['parent_manifest']['path']); training = read(job['matched_element_training']['path'])
        scales = np.array([x['scale'] for x in training['trace']], dtype=np.float64)
        default = float(manifest['spec']['default_scale'])
        if not np.isfinite(scales).all() or np.any(scales <= 0) or not np.isfinite(default) or default <= 0:
            raise ValueError('invalid physical scale')
        quantiles = np.quantile(scales, [0., .1, .5, .9, 1.]).tolist()
        rows.append(dict(key=job['key'], family=job['family'], default_scale=default,
            source_quantiles=quantiles, source_fraction_strictly_below_default=float(np.mean(scales < default)),
            default_inside_source_10_90=bool(quantiles[1] <= default <= quantiles[3]),
            training=job['matched_element_training'], parent_manifest=job['parent_manifest']))
    if len(rows) != 30: raise ValueError('incomplete scale audit')
    ranks = [r['source_fraction_strictly_below_default'] for r in rows]
    write(ROOT / 'diagnostics/source_default_scales.json', dict(fields=30, records=rows,
        all_defaults_inside_source_10_90=all(r['default_inside_source_10_90'] for r in rows),
        default_rank_range=[min(ranks), max(ranks)], quantile_levels=[0., .1, .5, .9, 1.],
        no_fitting=True, benchmark_queries_used=False, ground_truth_used=False,
        completed_before_new_benchmark_prediction=not (ROOT / 'predictions_complete.json').exists(),
        interpretation='Defaults lie within the source scale distribution; this does not establish correct object-specific query scales or inspect explicit overrides.',
        code=receipt(Path(__file__))))
    print('source default-scale audit', len(rows), [min(ranks), max(ranks)])


if __name__ == '__main__': main()
