"""Full-cohort known-support diagnosis for the joint conditional repair."""
import numpy as np
import torch

from scripts.research import diagnose_query_field_observation_mass as mass
from scripts.research.run_query_field_joint_elements import ROOT, CONTROL, ARMS, read, write, receipt
from scripts.research.run_query_field_protected import ROOT as RENDERED


def main():
    mass.main(ROOT, ARMS[0], dict(
        protected_training=RENDERED / 'protected_training/predictions/uqis',
        element_relation=CONTROL / 'element_relation/predictions/uqis',
        conditional_relation=CONTROL / 'conditional_relation/predictions/uqis'))
    directory = ROOT / 'diagnostics/point_observation_mass'; report = read(directory / 'report.json')
    comparisons, known = {}, {}
    for arm in report['summary']:
        clusters, scenes, cases = [], [], []
        for scene in report['scenes']:
            values = [q['arms'][arm] for q in scene['queries'] if q['arms'][arm]['correct_registered_support']]
            cases.extend(values)
            paired = [q['strata'][4]['recall'] - q['strata'][0]['recall'] for q in values
                      if q['strata'][4]['recall'] is not None and q['strata'][0]['recall'] is not None]
            if paired:
                clusters.append(paired); scenes.append(dict(scene=scene['scene'], queries=len(paired), delta=float(np.mean(paired))))
        rng = np.random.default_rng(20261008)
        bootstrap = [float(np.mean([v for i in ids for v in clusters[i]]))
                     for ids in rng.integers(0, len(clusters), (2000, len(clusters)))]
        values = [v for cluster in clusters for v in cluster]
        comparisons[arm] = dict(high_minus_low_query_mean=float(np.mean(values)), queries=len(values),
            lower=float(np.quantile(bootstrap, .025)), upper=float(np.quantile(bootstrap, .975)), per_scene=scenes)
        known[arm] = dict(queries=len(cases), **{k: float(np.mean([q[k] for q in cases]))
            for k in ['observed_target_recall', 'all_target_recall', 'actual_iou', 'actual_precision']})
    write(directory / 'within_query_comparison.json', dict(report=receipt(directory / 'report.json'),
        comparison=comparisons, known_support=known, bootstrap_samples=2000, seed=20261008,
        causal_claim=False, source_contribution_is_training_gradient=False, model_selection=False,
        aggregation='paired high-minus-low within queries; physical scenes resampled as clusters',
        code=receipt(__file__)))
    print('known support', known, flush=True)
    print('exposure gap', comparisons[ARMS[0]], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
