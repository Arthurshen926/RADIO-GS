"""Post-score known-support attribution, including weak-exposure target recall."""
import numpy as np
import torch

from scripts.research import diagnose_query_field_observation_mass as mass
from scripts.research.run_query_field_elements import ROOT, CONTROL, read, write, receipt


def main():
    mass.main(ROOT, 'conditional_relation', dict(
        protected_training=CONTROL / 'protected_training/predictions/uqis',
        element_relation=ROOT / 'element_relation/predictions/uqis'))
    directory = ROOT / 'diagnostics/point_observation_mass'
    report = read(directory / 'report.json')
    arms = list(report['summary']); comparisons = {}
    for arm in arms:
        clusters, counts = [], []
        for scene in report['scenes']:
            cases = [q['arms'][arm] for q in scene['queries'] if q['arms'][arm]['correct_registered_support']]
            paired = [q['strata'][4]['recall'] - q['strata'][0]['recall'] for q in cases
                      if q['strata'][4]['recall'] is not None and q['strata'][0]['recall'] is not None]
            if paired:
                clusters.append(paired); counts.append(dict(scene=scene['scene'], queries=len(paired), delta=float(np.mean(paired))))
        rng = np.random.default_rng(20261008)
        boot = [float(np.mean([x for i in ids for x in clusters[i]]))
                for ids in rng.integers(0, len(clusters), (2000, len(clusters)))]
        values = [x for cluster in clusters for x in cluster]
        comparisons[arm] = dict(high_minus_low_query_mean=float(np.mean(values)), queries=len(values),
            lower=float(np.quantile(boot, .025)), upper=float(np.quantile(boot, .975)), per_scene=counts)
    known_support = {}
    for arm in arms:
        cases = [q['arms'][arm] for s in report['scenes'] for q in s['queries'] if q['arms'][arm]['correct_registered_support']]
        known_support[arm] = dict(queries=len(cases), **{k: float(np.mean([q[k] for q in cases]))
            for k in ['observed_target_recall', 'all_target_recall', 'actual_iou', 'actual_precision']})
    write(directory / 'within_query_comparison.json', dict(
        report=receipt(directory / 'report.json'), comparison=comparisons, known_support=known_support,
        bootstrap_samples=2000, seed=20261008, causal_claim=False,
        aggregation='paired high-minus-low within queries; physical scenes resampled as clusters',
        source_contribution_is_training_gradient=False, model_selection=False))
    print('known support', known_support, flush=True)
    print('within-query exposure gaps', comparisons, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
