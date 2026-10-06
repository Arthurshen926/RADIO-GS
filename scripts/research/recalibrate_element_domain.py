"""Repair decision calibration without changing the independently fitted model."""
import json
from pathlib import Path
import joblib
import numpy as np
from fit_conditional_element_membership import evaluate


def main():
    source=Path('output/learned_instance_additions')
    root=Path('output/evidence_convergence_20261005');root.mkdir(parents=True,exist_ok=True)
    state=joblib.load(source/'conditional_query_elements.joblib')
    state['estimator'].n_jobs=4;state['region_estimator'].n_jobs=4
    design=json.loads((source/'design.json').read_text());thresholds=np.arange(.05,1,.05)
    rows,macro=evaluate(source,design['calibration'],state['region_estimator'],state['estimator'],thresholds,
        local=True,anchor_condition=True,joint_observations=True,query_region=True)
    old=float(state['threshold']);chosen=float(thresholds[np.argmax(macro[:,0])])
    state.update(threshold=chosen,calibration_selection_domain='complete deployment candidate domain',
        unlabelled_training_vertices_excluded=True,calibration_benchmark_labels_used=False)
    joblib.dump(state,root/'conditional_query_elements.joblib',compress=3)
    validation,metrics=evaluate(source,design['validation']+['scene0548_00','scene0079_00','scene0371_00','scene0536_00'],
        state['region_estimator'],state['estimator'],[old,chosen],
        local=True,anchor_condition=True,joint_observations=True,query_region=True)
    report=dict(original_threshold=old,threshold=chosen,thresholds=thresholds.tolist(),calibration=rows,
        calibration_macro=macro.tolist(),validation=validation,validation_macro=metrics.tolist(),
        training_unchanged=True,selection_domain='all inference candidates, including unlabelled vertices',
        benchmark_labels_used=False,validation_scenes_previously_opened=True)
    (root/'element_domain_calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL',old,chosen,metrics.tolist(),flush=True)


if __name__=='__main__':main()
